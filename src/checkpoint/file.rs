//! Memory-mapped safetensors with validated headers and indexed tensor spans.
//! Tensor views borrow from the mapping.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hrx::artifacts::safetensors::{DType, FileView};

use super::{Error, Result};

/// One tensor's dtype, shape and bytes.
#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a> {
    /// As safetensors names it: `I8`, `F32`, `BF16`, `F8_E4M3`.
    pub dtype: &'a str,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

impl Tensor<'_> {
    pub fn rows(&self) -> Result<usize> {
        self.shape.first().copied().ok_or_else(|| Error("a tensor with no shape".into()))
    }

    /// Bytes per row, for the two-dimensional weight tensors.
    pub fn row_bytes(&self) -> Result<usize> {
        let rows = self.rows()?;
        if rows == 0 || !self.bytes.len().is_multiple_of(rows) {
            return Err(Error("tensor size is not a whole number of rows".into()));
        }
        Ok(self.bytes.len() / rows)
    }
}

struct Entry {
    dtype: &'static str,
    shape: Vec<usize>,
    original: String,
}

/// The dtype names this runtime matches on, which are safetensors' own.
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::BOOL => "BOOL",
        DType::U8 => "U8",
        DType::I8 => "I8",
        DType::F8_E4M3 => "F8_E4M3",
        DType::F8_E5M2 => "F8_E5M2",
        DType::I16 => "I16",
        DType::U16 => "U16",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::I32 => "I32",
        DType::U32 => "U32",
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::I64 => "I64",
        DType::U64 => "U64",
        // Something this runtime has never seen. Every consumer matches on the
        // names above and reports the rest as unsupported, which this is.
        _ => "UNSUPPORTED",
    }
}

/// A tensor every transformer checkpoint carries, whatever wraps its names.
const ANCHOR: &str = "blocks.0.attn.wq.weight";

/// Removes the namespace a whole-model save puts around the transformer (ComfyUI's
/// `ModelSave` writes its state dict under the wrapper's attribute path). The
/// prefix is whatever precedes [`ANCHOR`], and only a prefix every tensor shares
/// is removed, so text encoders and VAEs keep their names.
fn unwrap_model(entries: BTreeMap<String, Entry>) -> BTreeMap<String, Entry> {
    let prefix = match entries.keys().find_map(|name| name.strip_suffix(ANCHOR)) {
        Some(prefix) if !prefix.is_empty() && prefix.ends_with('.') => prefix.to_string(),
        _ => return entries,
    };
    if !entries.keys().all(|name| name.starts_with(&prefix)) {
        return entries;
    }
    entries.into_iter().map(|(name, entry)| (name[prefix.len()..].to_string(), entry)).collect()
}

pub struct Checkpoint {
    path: PathBuf,
    file: FileView,
    entries: BTreeMap<String, Entry>,
}

impl std::fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Checkpoint({}, {} tensors)", self.path.display(), self.entries.len())
    }
}

impl Checkpoint {
    /// Maps a `.safetensors` file and indexes it.
    pub fn open(path: &Path) -> Result<Checkpoint> {
        if path.extension().is_none_or(|e| e != "safetensors") {
            return Err(Error(format!(
                "{} is not a ComfyUI checkpoint (.safetensors)",
                path.display()
            )));
        }
        // Safety: model checkpoints are opened read-only and must remain immutable
        // and untruncated while the session retains their mapping.
        #[allow(unsafe_code)]
        let file = unsafe { FileView::map(path) }
            .map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
        let entries = file
            .entries()
            .iter()
            .map(|(name, entry)| {
                (
                    name.clone(),
                    Entry {
                        dtype: dtype_name(entry.dtype),
                        shape: entry.shape.clone(),
                        original: name.clone(),
                    },
                )
            })
            .collect();
        let entries = unwrap_model(entries);
        Ok(Checkpoint { path: path.to_path_buf(), file, entries })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<Tensor<'_>> {
        let entry = self.entries.get(name).ok_or_else(|| {
            Error(format!("missing tensor {name} in {}", self.path.display()))
        })?;
        Ok(Tensor {
            dtype: entry.dtype,
            shape: &entry.shape,
            bytes: self
                .file
                .get(&entry.original)
                .map_err(|error| Error(error.to_string()))?
                .bytes,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// The transformer blocks the file carries, counted by their attention
    /// weights.
    pub fn block_count(&self) -> usize {
        (0..).take_while(|i| self.has(&format!("blocks.{i}.attn.wq.weight"))).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a safetensors file in a temporary directory.
    fn write(name: &str, header: &str, payload: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("krea2-{}-{name}", std::process::id()));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        std::fs::write(&path, bytes).expect("writing the fixture");
        path
    }

    #[test]
    fn a_file_that_is_not_safetensors_is_refused_by_name() {
        let error = Checkpoint::open(Path::new("/models/krea2.bin")).unwrap_err();
        assert!(error.0.contains("not a ComfyUI checkpoint (.safetensors)"), "{error}");
    }

    #[test]
    fn a_header_longer_than_the_file_is_refused() {
        let path = write("long.safetensors", "{}", b"");
        let mut bytes = std::fs::read(&path).expect("the fixture");
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, bytes).expect("rewriting the fixture");
        let error = Checkpoint::open(&path).unwrap_err();
        assert!(error.0.starts_with("cannot read "), "{error}");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_tensor_reaching_past_the_data_is_refused() {
        let header = r#"{"w":{"dtype":"I8","shape":[2,2],"data_offsets":[0,64]}}"#;
        let path = write("span.safetensors", header, &[0u8; 4]);
        let error = Checkpoint::open(&path).unwrap_err();
        assert!(error.0.starts_with("cannot read "), "{error}");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn tensors_read_back_with_their_dtype_shape_and_bytes() {
        let header = r#"{"__metadata__":{"format":"pt"},
            "w":{"dtype":"I8","shape":[2,3],"data_offsets":[0,6]},
            "s":{"dtype":"F32","shape":[2],"data_offsets":[6,14]}}"#;
        let path =
            write("read.safetensors", header, &[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0]);
        let file = Checkpoint::open(&path).expect("the fixture opens");
        let w = file.get("w").expect("w");
        assert_eq!((w.dtype, w.shape, w.bytes), ("I8", &[2, 3][..], &[1, 2, 3, 4, 5, 6][..]));
        assert_eq!((w.rows().unwrap(), w.row_bytes().unwrap()), (2, 3));
        assert_eq!(file.get("s").expect("s").dtype, "F32");
        assert!(!file.has("__metadata__"), "metadata is not a tensor");
        let error = file.get("missing").unwrap_err();
        assert!(error.0.starts_with("missing tensor missing in "), "{error}");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_whole_model_save_reads_under_the_transformer_names() {
        let header = r#"{"__metadata__":{"workflow":"{}"},
            "model.diffusion_model.blocks.0.attn.wq.weight":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},
            "model.diffusion_model.first.weight":{"dtype":"I8","shape":[1],"data_offsets":[1,2]}}"#;
        let path = write("wrapped.safetensors", header, &[7, 9]);
        let file = Checkpoint::open(&path).expect("the fixture opens");
        assert_eq!(file.get("first.weight").expect("unwrapped").bytes, &[9]);
        assert_eq!(file.block_count(), 1);
        assert!(!file.has("model.diffusion_model.first.weight"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn names_without_a_shared_wrapper_are_kept() {
        let header = r#"{"model.blocks.0.attn.wq.weight":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},
            "first.weight":{"dtype":"I8","shape":[1],"data_offsets":[1,2]}}"#;
        let path = write("mixed.safetensors", header, &[7, 9]);
        let file = Checkpoint::open(&path).expect("the fixture opens");
        assert!(file.has("model.blocks.0.attn.wq.weight") && file.has("first.weight"));
        std::fs::remove_file(path).ok();
    }
}
