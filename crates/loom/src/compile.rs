//! Running `loom-compile`, and the lock that lets several processes share a cache.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{Error, Result, Settings};

/// Resolves the compiler: explicit override, `LOOM_COMPILE`, runtime cache,
/// then `loom-compile` on PATH.
pub fn compiler(override_path: Option<&str>) -> String {
    resolve_compiler(
        override_path,
        std::env::var("LOOM_COMPILE").ok().as_deref(),
        runtime_directory().as_deref(),
    )
}

/// Compiler selection with explicit inputs for isolated tests.
fn resolve_compiler(
    override_path: Option<&str>,
    environment: Option<&str>,
    cache: Option<&Path>,
) -> String {
    if let Some(path) = override_path {
        return path.to_string();
    }
    if let Some(named) = environment {
        return named.to_string();
    }
    if let Some(cached) = cache.map(|root| root.join("loom-compile")) {
        if cached.is_file() {
            return cached.to_string_lossy().into_owned();
        }
    }
    // Nothing named it and nothing cached it: PATH is the last word.
    "loom-compile".to_string()
}

/// Optional runtime installation under the user's kernel-cache root.
pub fn runtime_directory() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(base.join("krea2-loom/runtime"))
}

/// One compilation: source text in, HSACO on disk out.
pub struct Compilation<'a> {
    pub compiler: &'a str,
    /// The source's name, which is also the kernel's symbol and the prefix its
    /// configuration keys take.
    pub name: &'a str,
    pub source: &'a str,
    /// String-valued because block kernels configure floats (`eps`, `scale`)
    /// as well as counts.
    pub config: &'a Settings,
}

impl Compilation<'_> {
    /// Compiles into `output`, using `scratch` for the source and the log.
    ///
    /// Shape settings are passed as `--config=krea2.<name>.<key>=<value>`.
    pub fn run(&self, scratch: &Path, output: &Path) -> Result<()> {
        let source_path = scratch.join(format!("{}.loom", self.name));
        let log_path = scratch.join(format!("{}.log", self.name));
        std::fs::write(&source_path, self.source)
            .map_err(|e| Error(format!("cannot write {}: {e}", source_path.display())))?;
        let mut command = Command::new(self.compiler);
        command
            .arg(&source_path)
            .arg("--backend=amdgpu-hal")
            .arg("--target=gfx1151")
            .arg(format!("--root=@krea2_{}", self.name))
            .arg(format!("--output={}", output.display()));
        for (key, value) in self.config {
            command.arg(format!("--config=krea2.{}.{key}={value}", self.name));
        }
        let log = std::fs::File::create(&log_path)
            .map_err(|e| Error(format!("cannot write {}: {e}", log_path.display())))?;
        let status = command.stderr(log).status().map_err(|e| {
            Error(format!("cannot start Loom compiler '{}': {e}", self.compiler))
        })?;
        if status.success() {
            return Ok(());
        }
        // The tail of the log is what makes a bad source or configuration
        // diagnosable through the C ABI's error string.
        let text = std::fs::read_to_string(&log_path).unwrap_or_default();
        let tail = &text[text.len().saturating_sub(600)..];
        Err(Error(format!(
            "Loom compilation failed: {}{}",
            self.name,
            if tail.is_empty() { String::new() } else { format!("\n{tail}") }
        )))
    }
}

/// An exclusive `flock` on `<directory>/.lock`. The descriptor is held only so
/// that dropping the value closes it, which is what releases the lock.
pub struct Lock(#[allow(dead_code)] std::fs::File);

impl Lock {
    pub fn acquire(directory: &Path, what: &str) -> Result<Lock> {
        let path = directory.join(".lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| Error(format!("cannot open {what} lock {}: {e}", path.display())))?;
        // Safety: flock takes a live descriptor; the file is held by `Lock`.
        let locked = unsafe {
            use std::os::fd::AsRawFd;
            extern "C" {
                fn flock(fd: i32, operation: i32) -> i32;
            }
            const LOCK_EX: i32 = 2;
            flock(file.as_raw_fd(), LOCK_EX) == 0
        };
        if !locked {
            return Err(Error(format!("cannot lock {what}")));
        }
        Ok(Lock(file))
    }
}

/// Files to remove when a compilation finishes, however it finishes.
pub struct Scratch(pub Vec<PathBuf>);

impl Drop for Scratch {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// `$XDG_CACHE_HOME`, else `~/.cache`, else `/tmp`, then `krea2-loom/<leaf>`.
pub fn user_cache_directory(leaf: &str) -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let path = base.join("krea2-loom").join(leaf);
    std::fs::create_dir_all(&path)
        .map_err(|e| Error(format!("cannot create {}: {e}", path.display())))?;
    Ok(path)
}

/// Hex SHA-256, the cache key everywhere in this crate.
pub fn digest(text: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text);
    hasher.finalize().iter().fold(String::with_capacity(64), |mut text, byte| {
        let _ = write!(text, "{byte:02x}");
        text
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_matches_the_reference_vectors() {
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn the_compiler_follows_the_caller_then_the_environment_then_the_cache() {
        // A cache with a compiler in it, and one without, both under this
        // test's control: reading the real environment would make the result
        // depend on whether this machine has an installed build.
        let root = std::env::temp_dir().join(format!("krea2-compiler-{}", std::process::id()));
        let populated = root.join("populated");
        let empty = root.join("empty");
        std::fs::create_dir_all(&populated).expect("a cache directory");
        std::fs::create_dir_all(&empty).expect("a cache directory");
        std::fs::write(populated.join("loom-compile"), b"").expect("a cached compiler");

        // The caller's choice wins over everything.
        assert_eq!(
            resolve_compiler(Some("/opt/loom-compile"), Some("/from/env"), Some(&populated)),
            "/opt/loom-compile"
        );
        // Then the environment, over a cache that has one.
        assert_eq!(resolve_compiler(None, Some("/from/env"), Some(&populated)), "/from/env");
        // Then the cache: this is what makes an installed build work, since
        // nothing puts loom-compile on PATH.
        assert_eq!(
            resolve_compiler(None, None, Some(&populated)),
            populated.join("loom-compile").to_string_lossy()
        );
        // A cache without one, and no cache at all, both fall through to PATH.
        assert_eq!(resolve_compiler(None, None, Some(&empty)), "loom-compile");
        assert_eq!(resolve_compiler(None, None, None), "loom-compile");
        std::fs::remove_dir_all(root).ok();
    }
}
