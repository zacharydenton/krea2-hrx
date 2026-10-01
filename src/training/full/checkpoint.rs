//! Bounded streaming checkpoints; the trainer publishes the enclosing directory atomically.
use super::{parameters::Parameters, spec};
use crate::{
    Error, Result,
    checkpoint::{Checkpoint, DType},
    lora::io,
};
use hrx::{Buffer, Stream};
use std::{collections::BTreeMap, fs::File, io::Write, path::Path};

const CHUNK: usize = 64 << 20;

/// Publish only a complete, synced checkpoint; never prune or overwrite old state.
pub(crate) fn publish(path: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    if path.exists() {
        return Err(Error::invalid(format!("checkpoint already exists: {}", path.display())));
    }
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name =
        path.file_name().ok_or_else(|| Error::invalid("checkpoint name"))?.to_string_lossy();
    let temporary = parent.join(format!(".{name}-{}.tmp", std::process::id()));
    std::fs::create_dir(&temporary).map_err(io)?;
    let result = (|| {
        write(&temporary)?;
        File::open(&temporary).map_err(io)?.sync_all().map_err(io)?;
        rustix::fs::renameat_with(
            rustix::fs::CWD,
            &temporary,
            rustix::fs::CWD,
            path,
            rustix::fs::RenameFlags::NOREPLACE,
        )
        .map_err(io)?;
        File::open(parent).map_err(io)?.sync_all().map_err(io)
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&temporary);
    }
    result
}
struct Entry<'a> {
    dtype: &'static str,
    shape: Vec<usize>,
    buffer: &'a Buffer,
    bytes: usize,
}

/// Needed payload bytes, excluding small JSON headers.
pub(crate) fn bytes() -> usize {
    spec::inventory()
        .values()
        .map(|p| if p.small { p.count * 12 } else { p.count * 4 + p.count.div_ceil(256) * 8 })
        .sum()
}

pub(crate) fn preflight(path: &Path) -> Result<()> {
    require_free(path, bytes())
}

pub(crate) fn require_free(path: &Path, payload: usize) -> Result<()> {
    let stats = rustix::fs::statvfs(path).map_err(io)?;
    let available = u128::from(stats.f_bavail) * u128::from(stats.f_frsize);
    let needed = payload as u128 + (2u128 << 30);
    if available < needed {
        return Err(Error::invalid(format!(
            "full checkpoint needs {:.1} GiB free, {:.1} GiB available at {} (existing checkpoints are retained until publication)",
            needed as f64 / (1u64 << 30) as f64,
            available as f64 / (1u64 << 30) as f64,
            path.display()
        )));
    }
    Ok(())
}

fn stream_file(
    path: &Path,
    entries: &BTreeMap<String, Entry<'_>>,
    stream: &mut Stream,
) -> Result<super::artifacts::Digest> {
    let specs: Vec<_> = entries
        .iter()
        .map(|(name, e)| (name.as_str(), e.dtype, e.shape.as_slice(), e.bytes))
        .collect();
    let mut file = super::artifacts::DigestWriter::new(File::create_new(path).map_err(io)?);
    write_header(&mut file, &specs)?;
    let mut staging =
        vec![0u8; CHUNK.min(entries.values().map(|e| e.bytes).max().unwrap_or(0))];
    for e in entries.values() {
        for offset in (0..e.bytes).step_by(CHUNK) {
            let n = (e.bytes - offset).min(CHUNK);
            stream.read_blocking(e.buffer.try_slice(offset, n)?, &mut staging[..n])?;
            validate_values(e.dtype, &staging[..n], false)?;
            file.write_all(&staging[..n]).map_err(io)?;
        }
    }
    file.finish()
}

pub(crate) fn write_header(
    out: &mut impl Write,
    entries: &[(&str, &str, &[usize], usize)],
) -> Result<()> {
    let mut header = BTreeMap::new();
    let mut offset = 0usize;
    for &(name, dtype, shape, bytes) in entries {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| Error::invalid("checkpoint offset overflow"))?;
        header.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,end]}),
        );
        offset = end;
    }
    let mut json = serde_json::to_vec(&header).map_err(io)?;
    json.resize(json.len().next_multiple_of(8), b' ');
    out.write_all(&(json.len() as u64).to_le_bytes()).map_err(io)?;
    out.write_all(&json).map_err(io)
}

pub(crate) fn validate_values(dtype: &str, bytes: &[u8], nonnegative: bool) -> Result<()> {
    let valid = match dtype {
        "U8" => true,
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| crate::numerics::to_f32(u16::from_le_bytes(*b)).is_finite()),
        "F32" => bytes.as_chunks::<4>().0.iter().all(|b| {
            let v = f32::from_le_bytes(*b);
            v.is_finite() && (!nonnegative || v >= 0.0)
        }),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(Error::invalid("nonfinite or invalid full-training checkpoint values"))
    }
}

impl Parameters {
    pub(crate) fn save_model(
        &self,
        stream: &mut Stream,
        directory: &Path,
    ) -> Result<super::artifacts::Digest> {
        self.save_files(stream, directory, false)
    }

    pub(crate) fn save(
        &self,
        stream: &mut Stream,
        directory: &Path,
    ) -> Result<super::artifacts::Digest> {
        self.save_files(stream, directory, true)
    }

    fn save_files(
        &self,
        stream: &mut Stream,
        directory: &Path,
        resumable: bool,
    ) -> Result<super::artifacts::Digest> {
        let mut model = BTreeMap::new();
        let mut optimizer = BTreeMap::new();
        for (name, p) in &self.values {
            let (dtype, buffer, bytes) = if let Some(master) = &p.master {
                ("F32", master.as_ref(), p.spec.count * 4)
            } else {
                ("BF16", p.value.as_ref(), p.spec.count * 2)
            };
            model.insert(
                name.clone(),
                Entry { dtype, shape: p.spec.shape.clone(), buffer, bytes },
            );
            for (kind, b) in [("first", &p.first), ("second", &p.second)] {
                optimizer.insert(
                    format!("{name}.{kind}"),
                    Entry {
                        dtype: if p.spec.small { "F32" } else { "U8" },
                        shape: p.spec.shape.clone(),
                        buffer: b,
                        bytes: p.spec.count * if p.spec.small { 4 } else { 1 },
                    },
                );
            }
            if let Some(b) = &p.scales {
                let parts = p.spec.count.div_ceil(256);
                optimizer.insert(
                    format!("{name}.scales"),
                    Entry { dtype: "F32", shape: vec![2, parts], buffer: b, bytes: parts * 8 },
                );
            }
        }
        require_free(
            directory,
            model.values().map(|e| e.bytes).sum::<usize>()
                + if resumable { optimizer.values().map(|e| e.bytes).sum() } else { 0 },
        )?;
        let model = stream_file(&directory.join("model.safetensors"), &model, stream)?;
        if resumable {
            stream_file(&directory.join("optimizer.safetensors"), &optimizer, stream)?;
        }
        Ok(model)
    }

    pub(crate) fn restore(&self, stream: &mut Stream, path: &Path) -> Result<()> {
        let file = Checkpoint::open(path)?;
        let expected: usize =
            self.values.values().map(|p| if p.spec.small { 2 } else { 3 }).sum();
        if file.names().count() != expected {
            return Err(Error::invalid("full optimizer tensor count mismatch"));
        }
        for (name, p) in &self.values {
            for (kind, buffer) in [
                ("first", Some(&p.first)),
                ("second", Some(&p.second)),
                ("scales", p.scales.as_ref()),
            ] {
                let Some(buffer) = buffer else { continue };
                let t = file.get(&format!("{name}.{kind}"))?;
                let dtype =
                    if p.spec.small || kind == "scales" { DType::F32 } else { DType::U8 };
                let shape = if kind == "scales" {
                    vec![2, p.spec.count.div_ceil(256)]
                } else {
                    p.spec.shape.clone()
                };
                if t.dtype != dtype || t.shape != shape {
                    return Err(Error::invalid("full optimizer shape/dtype mismatch"));
                }
                for (part, chunk) in t.bytes.chunks(CHUNK).enumerate() {
                    validate_values(
                        if dtype == DType::F32 { "F32" } else { "U8" },
                        chunk,
                        kind != "first",
                    )?;
                    stream.upload(buffer.try_slice(part * CHUNK, chunk.len())?, chunk)?;
                }
            }
            stream.synchronize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_checkpoint_write_preserves_previous_complete_state() {
        let dir = tempfile::tempdir().unwrap();
        let old = dir.path().join("step-000001");
        std::fs::create_dir(&old).unwrap();
        std::fs::write(old.join("state.json"), b"previous").unwrap();
        let next = dir.path().join("step-000002");
        let error = publish(&next, |temp| {
            std::fs::write(temp.join("model.safetensors"), b"complete").map_err(io)?;
            std::fs::write(temp.join("optimizer.safetensors"), b"partial").map_err(io)?;
            Err(Error::invalid("injected disk write failure"))
        });
        assert!(error.is_err());
        assert!(!next.exists());
        assert_eq!(std::fs::read(old.join("state.json")).unwrap(), b"previous");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        publish(&next, |temp| std::fs::write(temp.join("state.json"), b"complete").map_err(io))
            .unwrap();
        assert_eq!(std::fs::read(next.join("state.json")).unwrap(), b"complete");
        assert!(publish(&next, |_| panic!("must not overwrite")).is_err());
    }
    #[test]
    fn streaming_header_roundtrip_and_failed_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let mut f = File::create(&path).unwrap();
        write_header(&mut f, &[("a", "BF16", &[2], 4), ("b", "F32", &[1], 4)]).unwrap();
        f.write_all(&[0, 0, 128, 63, 0, 0, 0, 64]).unwrap();
        drop(f);
        let c = Checkpoint::open(&path).unwrap();
        assert_eq!(c.get("a").unwrap().bytes, &[0, 0, 128, 63]);
        assert_eq!(c.get("b").unwrap().bytes, &2f32.to_le_bytes());
        struct Fail;
        impl Write for Fail {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert!(write_header(&mut Fail, &[("a", "BF16", &[2], 4)]).is_err());
        assert!(validate_values("F32", &f32::NAN.to_le_bytes(), false).is_err());
        assert!(validate_values("F32", &(-1f32).to_le_bytes(), true).is_err());
    }
    #[test]
    fn publication_cannot_replace_a_destination_created_during_write() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("checkpoint");
        let result = publish(&out, |tmp| {
            std::fs::write(tmp.join("model"), b"new").map_err(io)?;
            std::fs::create_dir(&out).map_err(io)
        });
        assert!(result.is_err());
        assert!(out.is_dir());
        assert!(!out.join("model").exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
