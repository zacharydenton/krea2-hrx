//! The kernel sources, embedded from `kernels/` at build time.
include!(concat!(env!("OUT_DIR"), "/sources.rs"));

/// One auxiliary kernel's Loom source (`kernels/native/<name>.loom`).
pub fn auxiliary(name: &str) -> Option<&'static str> {
    AUXILIARY.iter().find(|(stem, _)| *stem == name).map(|(_, source)| *source)
}

/// One block kernel's Loom source (`kernels/<name>.loom`).
pub fn block(name: &str) -> Option<&'static str> {
    BLOCK.iter().find(|(stem, _)| *stem == name).map(|(_, source)| *source)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The library must carry exactly the repository's kernels: a stale build
    /// directory is otherwise invisible until an image changes.
    fn matches_disk(table: &[(&str, &str)], directory: &str) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(directory);
        assert!(!table.is_empty(), "no kernels embedded from {directory}");
        for (name, embedded) in table {
            let path = root.join(format!("{name}.loom"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            assert_eq!(*embedded, text, "{} is stale in the build", path.display());
        }
        let on_disk = std::fs::read_dir(&root)
            .expect("the kernel directory exists")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|e| e == "loom"))
            .count();
        assert_eq!(on_disk, table.len(), "{directory} has kernels the build did not embed");
    }

    #[test]
    fn embedded_auxiliary_kernels_match_the_repository() {
        matches_disk(AUXILIARY, "kernels/native");
    }

    #[test]
    fn embedded_block_kernels_match_the_repository() {
        matches_disk(BLOCK, "kernels");
    }

    #[test]
    fn the_kernels_the_host_launches_by_name_are_present() {
        for name in ["unary_one", "euler", "guidance", "im2col", "softmax"] {
            assert!(auxiliary(name).is_some(), "missing auxiliary kernel {name}");
        }
        for name in ["gemm_i8_256", "attention_gqa_lds_f16_wmma", "prepare_norm_i8"] {
            assert!(block(name).is_some(), "missing block kernel {name}");
        }
    }
}
