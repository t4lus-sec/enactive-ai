use std::fs;
use std::path::{Path, PathBuf};

const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;

pub struct ImageInput {
    caption: String,
    canonical_path: PathBuf,
    mime_type: &'static str,
    byte_length: usize,
    fingerprint: u64,
    data_url: String,
}

impl ImageInput {
    pub fn routing_text(&self) -> String {
        format!(
            "[Image attached: {} | {} | {} bytes]\n{}",
            self.file_name(),
            self.mime_type,
            self.byte_length,
            self.caption,
        )
    }

    /// `time`: the header's time lines, from `clock::Arrival::header`.
    pub fn working_content(&self, time: &str) -> String {
        format!(
            "[TURN RUNTIME STATE]\n\
             {}\n\n\
             [CURRENT INPUT]\n\n\
             {}\n\n\
             [ATTACHED IMAGE]\n\
             source_path={:?}\n\
             mime_type={}\n\
             byte_length={}\n\
             content_fingerprint=fnv1a64:{:016x}\n\
             The image bytes are attached to this message for the current agent cycle. Later turns retain this record and s0ma's response, but must reattach the file to inspect its pixels again.",
            time,
            self.caption,
            self.canonical_path.display().to_string(),
            self.mime_type,
            self.byte_length,
            self.fingerprint,
        )
    }

    pub fn memory_record(&self) -> String {
        format!(
            "[IMAGE ATTACHMENT]\n\
             source_path={:?}\n\
             mime_type={}\n\
             byte_length={}\n\
             content_fingerprint=fnv1a64:{:016x}\n\
             image_bytes_persisted=false\n\n\
             {}",
            self.canonical_path.display().to_string(),
            self.mime_type,
            self.byte_length,
            self.fingerprint,
            self.caption,
        )
    }

    pub fn into_data_url(self) -> String {
        self.data_url
    }

    fn file_name(&self) -> String {
        self.canonical_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.canonical_path.display().to_string())
    }
}

/// Parse and load `/img <path> <message/caption>`.
///
/// Paths may be unquoted, single-quoted, or double-quoted. Backslash escaping
/// is supported inside both quoted and unquoted paths. The caption is kept as
/// ordinary text after the first path argument.
pub fn parse_image_command(input: &str) -> Result<Option<ImageInput>, String> {
    let command = input.trim();

    let Some(rest) = command.strip_prefix("/img") else {
        return Ok(None);
    };

    if rest
        .chars()
        .next()
        .is_some_and(|character| !character.is_whitespace())
    {
        return Ok(None);
    }

    let (raw_path, caption) = split_path_and_caption(rest)?;

    if caption.trim().is_empty() {
        return Err("usage: /img <path> <message/caption>".to_string());
    }

    let path = expand_home(&raw_path);
    let canonical_path = fs::canonicalize(&path).map_err(|error| {
        format!(
            "cannot open image {:?}: {error}",
            path.display().to_string()
        )
    })?;
    let metadata = fs::metadata(&canonical_path).map_err(|error| {
        format!(
            "cannot inspect image {:?}: {error}",
            canonical_path.display().to_string()
        )
    })?;

    if !metadata.is_file() {
        return Err(format!(
            "image path is not a regular file: {:?}",
            canonical_path.display().to_string()
        ));
    }

    if metadata.len() == 0 {
        return Err("image file is empty".to_string());
    }

    if metadata.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "image is {} bytes; the uploader limit is {} bytes",
            metadata.len(),
            MAX_IMAGE_BYTES,
        ));
    }

    let bytes = fs::read(&canonical_path).map_err(|error| {
        format!(
            "cannot read image {:?}: {error}",
            canonical_path.display().to_string()
        )
    })?;
    let mime_type = detect_mime(&canonical_path, &bytes).ok_or_else(|| {
        "unsupported image format; use PNG, JPEG, GIF, BMP, WebP, or TGA".to_string()
    })?;
    let fingerprint = crate::util::fnv1a64(&bytes);
    let data_url = format!("data:{mime_type};base64,{}", encode_base64(&bytes));

    Ok(Some(ImageInput {
        caption: caption.trim().to_string(),
        canonical_path,
        mime_type,
        byte_length: bytes.len(),
        fingerprint,
        data_url,
    }))
}

fn split_path_and_caption(input: &str) -> Result<(String, &str), String> {
    let input = input.trim_start();

    if input.is_empty() {
        return Err("usage: /img <path> <message/caption>".to_string());
    }

    let mut characters = input.char_indices();
    let (_, first) = characters.next().expect("nonempty input checked above");
    let quoted = matches!(first, '\'' | '"');
    let quote = if quoted { Some(first) } else { None };
    let mut path = String::new();
    let mut escaped = !quoted && first == '\\';

    if !quoted && !escaped {
        path.push(first);
    }

    for (index, character) in characters {
        if escaped {
            path.push(character);
            escaped = false;
            continue;
        }

        if character == '\\' {
            escaped = true;
            continue;
        }

        if let Some(quote) = quote {
            if character == quote {
                let caption = &input[index + character.len_utf8()..];

                if path.is_empty() {
                    return Err("image path cannot be empty".to_string());
                }

                return Ok((path, caption));
            }

            path.push(character);
        } else if character.is_whitespace() {
            if path.is_empty() {
                return Err("image path cannot be empty".to_string());
            }

            return Ok((path, &input[index..]));
        } else {
            path.push(character);
        }
    }

    if escaped {
        path.push('\\');
    }

    if quote.is_some() {
        return Err("unterminated quoted image path".to_string());
    }

    if path.is_empty() {
        return Err("image path cannot be empty".to_string());
    }

    Ok((path, ""))
}

fn expand_home(path: &str) -> PathBuf {
    let Some(remainder) = path.strip_prefix("~/") else {
        return PathBuf::from(path);
    };

    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(remainder),
        None => PathBuf::from(path),
    }
}

fn detect_mime(path: &Path, bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("tga"))
    {
        Some("image/x-tga")
    } else {
        None
    }
}

/// Encode already-loaded image bytes for an OpenAI-compatible vision message.
/// The bytes remain transient; callers decide when to release the message.
pub fn image_data_url(path: &Path, bytes: &[u8]) -> Result<String, String> {
    let mime_type = detect_mime(path, bytes).ok_or_else(|| {
        "unsupported image format; use PNG, JPEG, GIF, BMP, WebP, or TGA".to_string()
    })?;

    Ok(format!("data:{mime_type};base64,{}", encode_base64(bytes)))
}


fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);

    for chunk in &mut chunks {
        let block = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32;

        encoded.push(TABLE[((block >> 18) & 0x3f) as usize] as char);
        encoded.push(TABLE[((block >> 12) & 0x3f) as usize] as char);
        encoded.push(TABLE[((block >> 6) & 0x3f) as usize] as char);
        encoded.push(TABLE[(block & 0x3f) as usize] as char);
    }

    match chunks.remainder() {
        [first] => {
            let block = (*first as u32) << 16;
            encoded.push(TABLE[((block >> 18) & 0x3f) as usize] as char);
            encoded.push(TABLE[((block >> 12) & 0x3f) as usize] as char);
            encoded.push('=');
            encoded.push('=');
        }
        [first, second] => {
            let block = ((*first as u32) << 16) | ((*second as u32) << 8);
            encoded.push(TABLE[((block >> 18) & 0x3f) as usize] as char);
            encoded.push(TABLE[((block >> 12) & 0x3f) as usize] as char);
            encoded.push(TABLE[((block >> 6) & 0x3f) as usize] as char);
            encoded.push('=');
        }
        [] => {}
        _ => unreachable!(),
    }

    encoded
}

#[cfg(test)]
mod tests {
    use super::{encode_base64, split_path_and_caption};

    #[test]
    fn base64_known_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn parses_quoted_path_and_caption() {
        let (path, caption) = split_path_and_caption("\"/tmp/a b.png\" explain this")
            .expect("quoted path should parse");

        assert_eq!(path, "/tmp/a b.png");
        assert_eq!(caption.trim(), "explain this");
    }

    #[test]
    fn parses_escaped_unquoted_path() {
        let (path, caption) = split_path_and_caption("/tmp/a\\ b.png explain this")
            .expect("escaped path should parse");

        assert_eq!(path, "/tmp/a b.png");
        assert_eq!(caption.trim(), "explain this");
    }
}
