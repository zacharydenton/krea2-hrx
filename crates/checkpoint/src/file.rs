//! A memory-mapped safetensors file.
//!
//! The format's own crate parses and validates the header; this adds the
//! mapping and an index of where each tensor's bytes are, so a tensor borrows
//! from the file it came from and cannot outlive it. The C++ handed out bare
//! pointers into the mapping and relied on discipline instead.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::{Dtype, SafeTensors};

use crate::{Error, Result};

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
    start: usize,
    end: usize,
}

/// The dtype names this runtime matches on, which are safetensors' own.
fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::BOOL => "BOOL",
        Dtype::U8 => "U8",
        Dtype::I8 => "I8",
        Dtype::F8_E4M3 => "F8_E4M3",
        Dtype::F8_E5M2 => "F8_E5M2",
        Dtype::I16 => "I16",
        Dtype::U16 => "U16",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::I32 => "I32",
        Dtype::U32 => "U32",
        Dtype::F32 => "F32",
        Dtype::F64 => "F64",
        Dtype::I64 => "I64",
        Dtype::U64 => "U64",
        // Something this runtime has never seen. Every consumer matches on the
        // names above and reports the rest as unsupported, which this is.
        _ => "UNSUPPORTED",
    }
}

pub struct Checkpoint {
    path: PathBuf,
    map: Mmap,
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
        let file = std::fs::File::open(path)
            .map_err(|e| Error(format!("cannot open {}: {e}", path.display())))?;
        // Safety: mapping a file is unsafe because another process truncating
        // it underneath us is undefined; the C++ host mapped the checkpoint the
        // same way, and this is a read-only model file.
        #[allow(unsafe_code)]
        let map = unsafe { Mmap::map(&file) }
            .map_err(|e| Error(format!("cannot map {}: {e}", path.display())))?;
        // The crate validates the header and every tensor's span; the index
        // keeps where each one lives so the views can be rebuilt per call
        // without reparsing 13 GB worth of header.
        let base = map.as_ptr() as usize;
        let entries = {
            let tensors = SafeTensors::deserialize(&map)
                .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
            tensors
                .tensors()
                .into_iter()
                .map(|(name, view)| {
                    let start = view.data().as_ptr() as usize - base;
                    let entry = Entry {
                        dtype: dtype_name(view.dtype()),
                        shape: view.shape().to_vec(),
                        start,
                        end: start + view.data().len(),
                    };
                    (name, entry)
                })
                .collect()
        };
        Ok(Checkpoint { path: path.to_path_buf(), map, entries })
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
            bytes: &self.map[entry.start..entry.end],
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
}
