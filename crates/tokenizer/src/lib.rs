//! Qwen3-VL's tokenizer, and Krea 2's prompt template.
//!
//! The tokenizer file is HuggingFace's own format, so this is the `tokenizers`
//! crate reading `assets/tokenizer.json`: the NFC normalization and the
//! byte-level pre-tokenizer come from the file rather than from code here,
//! which is why 5,000 lines of hand-written Unicode tables go away with it.
//! What is ours is the template, the truncation and the input limit.
use tokenizers::Tokenizer as Inner;

/// The tokenizer file, in the library, as it was in the C++ host.
pub const EMBEDDED: &[u8] = include_bytes!("../../../assets/tokenizer.json");

/// The longest prompt this accepts, in UTF-8 bytes.
const MAX_BYTES: usize = 65536;

/// Krea 2 conditions on this template; the model's own encoder strips the
/// system and user opening again, 34 tokens of it.
const PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, \
                      size, texture, quantity, text, spatial relationships of the objects and \
                      background:<|im_end|>\n<|im_start|>user\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// The template plus a prompt is truncated to this many tokens before the
/// assistant turn is appended.
const MAX_TEMPLATED: usize = 541;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub struct Tokenizer(Inner);

impl Tokenizer {
    /// The tokenizer embedded in this library.
    pub fn embedded() -> Result<Tokenizer> {
        Self::from_bytes(EMBEDDED)
    }

    pub fn from_bytes(json: &[u8]) -> Result<Tokenizer> {
        Inner::from_bytes(json)
            .map(Tokenizer)
            .map_err(|e| Error(format!("cannot read the tokenizer: {e}")))
    }

    pub fn from_file(path: &std::path::Path) -> Result<Tokenizer> {
        let json = std::fs::read(path)
            .map_err(|e| Error(format!("cannot read tokenizer: {}: {e}", path.display())))?;
        Self::from_bytes(&json)
    }

    /// Token ids for arbitrary text, with no template and no added specials —
    /// special tokens written out in the text itself are still recognized.
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        if text.len() > MAX_BYTES {
            return Err(Error(format!("prompt exceeds {MAX_BYTES} UTF-8 bytes")));
        }
        let encoding =
            self.0.encode(text, false).map_err(|e| Error(format!("cannot tokenize: {e}")))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Text for token ids, for tests and for anything that wants to show what
    /// the model was given.
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.0.decode(&ids, false).map_err(|e| Error(format!("cannot decode: {e}")))
    }

    /// The prompt under Krea's chat template, truncated as the model expects.
    pub fn prompt(&self, text: &str) -> Result<Vec<i32>> {
        let mut ids = self.encode(&format!("{PREFIX}{text}"))?;
        ids.truncate(MAX_TEMPLATED);
        ids.extend(self.encode(SUFFIX)?);
        Ok(ids)
    }
}

/// NFC, for comparing round-tripped text with its normalized input.
#[cfg(test)]
fn nfc_of(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfc().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokenizer {
        Tokenizer::embedded().expect("the embedded tokenizer")
    }

    #[test]
    fn the_embedded_file_is_the_repositorys() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("the crate sits in the repository")
            .join("assets/tokenizer.json");
        assert_eq!(EMBEDDED, std::fs::read(path).expect("the tokenizer file"));
    }

    #[test]
    fn the_template_costs_thirty_four_tokens_before_the_prompt() {
        // The model's text encoder drops exactly this many rows, so the count
        // is part of the contract rather than an implementation detail.
        let tokenizer = tokenizer();
        let prefix = tokenizer.encode(PREFIX).expect("the prefix");
        assert_eq!(prefix.len(), 34);
        assert_eq!(prefix[30..], [198, 151644, 872, 198], "…\\n<|im_start|>user\\n");
        let suffix = tokenizer.encode(SUFFIX).expect("the suffix");
        assert_eq!(suffix, [151645, 198, 151644, 77091, 198]);
    }

    #[test]
    fn a_prompt_is_the_template_around_it() {
        let tokenizer = tokenizer();
        let ids = tokenizer.prompt("a red fox in the snow").expect("a prompt");
        let text = tokenizer.encode("a red fox in the snow").expect("the text");
        assert_eq!(ids.len(), 34 + text.len() + 5);
        assert_eq!(&ids[34..34 + text.len()], &text[..]);
    }

    #[test]
    fn a_prompt_longer_than_the_model_takes_is_truncated_before_the_turn() {
        let tokenizer = tokenizer();
        let ids = tokenizer.prompt(&"word ".repeat(2000)).expect("a long prompt");
        assert_eq!(ids.len(), MAX_TEMPLATED + 5);
        assert_eq!(&ids[MAX_TEMPLATED..], &[151645, 198, 151644, 77091, 198]);
    }

    #[test]
    fn text_beyond_the_byte_limit_is_refused() {
        let error = tokenizer().encode(&"x".repeat(MAX_BYTES + 1)).unwrap_err();
        assert_eq!(error.0, "prompt exceeds 65536 UTF-8 bytes");
    }

    #[test]
    fn special_tokens_written_in_the_text_are_recognized() {
        let ids = tokenizer().encode("hello <|im_start|>world").expect("mixed text");
        assert!(ids.contains(&151644), "<|im_start|> should be one token: {ids:?}");
    }

    /// The file declares an NFC normalizer, and the model was trained behind
    /// it: composed and decomposed spellings must reach the same ids.
    #[test]
    fn the_normalizer_in_the_file_is_applied() {
        let tokenizer = tokenizer();
        for (composed, decomposed) in [
            ("café", "cafe\u{0301}"),
            ("각", "\u{1100}\u{1161}\u{11a8}"),
            ("q\u{0323}\u{0307}", "q\u{0307}\u{0323}"),
        ] {
            assert_eq!(
                tokenizer.encode(composed).expect("composed"),
                tokenizer.encode(decomposed).expect("decomposed"),
                "{composed:?} and {decomposed:?} normalize together"
            );
        }
    }

    /// Scripts, emoji, whitespace runs and combining marks all round-trip
    /// through the byte-level pre-tokenizer.
    #[test]
    fn mixed_script_text_encodes_and_decodes_back() {
        let tokenizer = tokenizer();
        for text in [
            "a red fox in the snow",
            "தமிழ் ௧௨௩",
            "   leading   and trailing  ",
            "\n \r\n",
            "中🦊 αß",
            "q\u{0307}\u{0323}",
        ] {
            let ids = tokenizer.encode(text).expect("mixed text");
            assert!(!ids.is_empty(), "{text:?} produced no tokens");
            let round_trip = tokenizer.decode(&ids).expect("decoding");
            assert_eq!(round_trip, super::nfc_of(text), "{text:?} did not round-trip");
        }
    }
}
