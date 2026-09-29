//! Offline compiler checks: no GPU device or stream is opened.
use krea2::kernels::{compiler, sources};

#[test]
#[ignore = "requires the provisioned Loom compiler, but no GPU execution"]
fn training_kernels_compile_without_a_device() {
    let compiler = compiler(None).unwrap();
    for &(name, source) in sources::AUXILIARY
        .iter()
        .filter(|(name, _)| name.starts_with("train_") || *name == "lora_transport")
    {
        let mut request = hrx::loom::Specialization::new(format!("krea2_{name}"));
        for (key, value) in [
            ("grid_x", 5),
            ("grid_y", 1),
            ("rows", 7),
            ("cols", 128),
            ("size", 896),
            ("tokens", 7),
            ("heads", 8),
            ("kv", 2),
            ("qsize", 7168),
            ("ksize", 1792),
            ("stats", 56),
            ("tables", 896),
            ("width", 8),
            ("parts", 1),
            ("reverse", 1),
        ] {
            request.set_config(format!("krea2.{name}.{key}"), value.to_string());
        }
        compiler.module(source).compile(&request).unwrap_or_else(|e| panic!("{name}: {e}"));
        if name == "lora_transport" {
            request.set_config("krea2.lora_transport.reverse", "2");
            compiler.module(source).compile(&request).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }
}
