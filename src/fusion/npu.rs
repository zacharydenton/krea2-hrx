//! Fixed native BF16 fusion projection with explicit GPU handoffs.
use super::{
    qualification::{self as q, Case, Record, Shape},
    FusionBackend,
};
use crate::ops::{Ops, Tensor, Weight};
use hrx::{
    execution::{
        Access, BindingContract, Buffer, ExecutableGraph, GpuAccess, KernelContract,
        MemoryPlacement, Runtime,
    },
    Error, Result, Stream,
};
use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    path::Path,
};

pub const STORAGE_LIMIT: usize = 512 << 20;
fn binding(bytes: usize, access: Access, layout: &str) -> BindingContract {
    BindingContract { bytes, alignment: 4, access, layout: layout.into() }
}
fn contract(shape: &Shape) -> KernelContract {
    KernelContract {
        bindings: vec![
            binding(shape.padded_m() * shape.k * 2, Access::Read, "row-major BF16 A"),
            binding(shape.n * shape.k * 2, Access::Read, "column-major BF16 B"),
            binding(shape.padded_m() * shape.n * 4, Access::Write, "row-major F32 C"),
        ],
        constants: vec![],
    }
}

/// Compile checked-in sources without opening an NPU. Returned records are
/// candidates; no automatic selection is allowed until qualification is complete.
pub fn compile(tools: &Path, case: Case) -> Result<Record> {
    if case.shape.storage_bytes().is_none_or(|bytes| bytes > STORAGE_LIMIT) {
        return Err(Error::Unsupported("fusion exceeds the pilot storage budget".into()));
    }
    let shape = &case.shape;
    let loom = hrx::loom::Compiler::resolve(None)?;
    let mut spec = hrx::loom::Specialization::new("krea2_fusion_epilogue");
    for (key, value) in [
        ("count", shape.m * shape.n),
        ("columns", shape.n),
        ("bias", usize::from(shape.bias)),
        ("grid_x", (shape.m * shape.n).div_ceil(256)),
    ] {
        spec.config.insert(format!("krea2.fusion_epilogue.{key}"), value.to_string());
    }
    let epilogue = loom
        .module(include_str!("../../kernels/native/fusion_epilogue.loom"))
        .compile(&spec)?;
    let toolchain = hrx::npu::compiler::Toolchain::load(tools)?;
    if !matches!(toolchain.backend, hrx::npu::compiler::Backend::Chess) {
        return Err(Error::Unsupported("fusion requires the Chess toolchain".into()));
    }
    let toolchain_identity = toolchain.identity()?;
    let compiler = hrx::npu::compiler::Compiler::new(
        toolchain,
        hrx::npu::compiler::CompilerOptions::new()?,
    )?;
    let artifact = compiler.compile(&hrx::npu::compiler::Project {
        root: Path::new(env!("CARGO_MANIFEST_DIR")).join("native/npu"),
        source: hrx::npu::compiler::Source::Iron("gemm.py".into()),
        dependencies: vec!["mm.cc".into(), "zero.cc".into()],
        arguments: vec![
            "--dev".into(),
            "npu2".into(),
            "-M".into(),
            shape.padded_m().to_string(),
            "-K".into(),
            shape.k.to_string(),
            "-N".into(),
            shape.n.to_string(),
            "--dtype_in".into(),
            "bf16".into(),
            "--dtype_out".into(),
            "f32".into(),
            "--use-chess".into(),
            "1".into(),
            "--b-col-maj".into(),
            "1".into(),
        ],
        cacheable: true,
        contract: contract(shape),
    })?;
    Ok(Record {
        schema: 1,
        implementation: q::implementation(),
        toolchain: toolchain_identity,
        machine: q::machine()?,
        artifact: artifact.path().into(),
        image: hrx::bundle::file_digest(&artifact.path().join("x.xclbin"))?,
        instructions: hrx::bundle::file_digest(&artifact.path().join("x.bin"))?,
        epilogue: epilogue.path().into(),
        epilogue_digest: hrx::bundle::file_digest(epilogue.path())?,
        case,
        processes: vec![],
        generation: None,
    })
}

pub struct Pilot {
    runtime: Runtime,
    input: Buffer,
    output: Buffer,
    graph: ExecutableGraph,
    pub bytes: usize,
    input_bytes: usize,
    output_bytes: usize,
}
impl Pilot {
    /// Load an artifact produced by this implementation's qualification command.
    /// # Safety
    /// The record and artifact store must be trusted. Hashes detect changes; they
    /// do not prove that arbitrary native code obeys the recorded memory contract.
    pub unsafe fn load(record: &Record, weights: &[u8], bias: &[u8]) -> Result<Self> {
        let shape = &record.case.shape;
        let bytes =
            shape.storage_bytes().filter(|bytes| *bytes <= STORAGE_LIMIT).ok_or_else(|| {
                Error::Unsupported("fusion exceeds the pilot storage budget".into())
            })?;
        if weights.len() != shape.n * shape.k * 2
            || bias.len() != if shape.bias { shape.n * 2 } else { 0 }
        {
            return Err(Error::Message(
                "fusion weight or bias extent differs from its shape".into(),
            ));
        }
        record.verify(
            shape,
            &record.case.checkpoint,
            &hrx::bundle::digest(weights),
            &hrx::bundle::digest(bias),
        )?;
        let runtime = Runtime::new()?;
        if runtime.gpu()?.target().as_str() != "gfx1151" {
            return Err(Error::Unsupported(
                "fusion pilot is qualified only on gfx1151/XDNA2".into(),
            ));
        }
        // The candidate identifies the checked-in generator, all emitted bytes,
        // exact argument extents and a supported fixed specialization.
        let program =
            unsafe { runtime.npu(0)?.load_program(record.artifact.join("x.xclbin")) }?;
        let kernel = unsafe {
            program.kernel(&fs::read(record.artifact.join("x.bin"))?, contract(shape))
        }?;
        let input = runtime.allocate(
            shape.padded_m() * shape.k * 2,
            MemoryPlacement::Shared(program.clone()),
        )?;
        let weight =
            runtime.allocate(weights.len(), MemoryPlacement::NpuLocal(program.clone()))?;
        weight.map_write()?.copy_from_slice(weights);
        let accumulation = runtime
            .allocate(shape.padded_m() * shape.n * 4, MemoryPlacement::Shared(program))?;
        let bias_buffer = runtime.allocate(shape.n * 2, MemoryPlacement::HostVisible)?;
        if shape.bias {
            bias_buffer.map_write()?.copy_from_slice(bias);
        }
        let output = runtime.allocate(shape.m * shape.n * 2, MemoryPlacement::GpuLocal)?;
        let stream = Stream::open()?;
        let raw = unsafe { stream.load(&record.epilogue, "krea2_fusion_epilogue") }?;
        let epilogue = unsafe {
            runtime.adopt_gpu_kernel(
                raw,
                [(shape.m * shape.n).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                KernelContract {
                    bindings: vec![
                        binding(shape.m * shape.n * 4, Access::Read, "F32"),
                        binding(shape.n * 2, Access::Read, "BF16 bias"),
                        binding(shape.m * shape.n * 2, Access::Write, "BF16"),
                    ],
                    constants: vec![],
                },
            )
        }?;
        let mut graph = runtime.graph();
        graph.npu(&kernel, &[input.view(), weight.view(), accumulation.view()])?;
        graph.gpu(&epilogue, &[accumulation.view(), bias_buffer.view(), output.view()])?;
        Ok(Self {
            runtime,
            input,
            output,
            graph: graph.prepare()?,
            bytes,
            input_bytes: shape.m * shape.k * 2,
            output_bytes: shape.m * shape.n * 2,
        })
    }
    pub fn execute(
        &self,
        stream: &mut Stream,
        input: hrx::View<'_>,
        output: hrx::View<'_>,
    ) -> Result<()> {
        if input.len() != self.input_bytes || output.len() != self.output_bytes {
            return Err(Error::Message(
                "fusion activation extent differs from its shape".into(),
            ));
        }
        // The callback only copies and queues work on this stream. Borrowed views
        // and native recordings never escape either handoff.
        unsafe {
            self.runtime.with_gpu_access(
                stream,
                &[GpuAccess { view: self.input.view(), access: Access::Write }],
                |stream, views| stream.copy(views[0].slice(0, input.len())?, input),
            )
        }?;
        self.graph.submit()?.wait()?;
        unsafe {
            self.runtime.with_gpu_access(
                stream,
                &[GpuAccess { view: self.output.view(), access: Access::Read }],
                |stream, views| stream.copy(output, views[0]),
            )
        }
    }
    pub fn statistics(&self) -> hrx::execution::Statistics {
        self.runtime.statistics()
    }
}

#[derive(Default)]
pub(crate) struct Cache {
    entries: VecDeque<(Shape, Option<Pilot>, String)>,
    captured: BTreeSet<String>,
    checkpoint: Option<String>,
    pub reason: String,
}
fn read(stream: &mut Stream, view: hrx::View<'_>) -> Result<Vec<u8>> {
    let mut bytes = vec![0; view.len()];
    stream.read_blocking(view, &mut bytes)?;
    Ok(bytes)
}
impl Cache {
    pub fn linear(
        &mut self,
        selection: (FusionBackend, &Path),
        stream: &mut Stream,
        ops: &Ops,
        x: &Tensor,
        w: &Weight,
        bias: Option<hrx::View<'_>>,
    ) -> crate::ops::Result<Option<Tensor>> {
        self.try_linear(selection, stream, ops, x, w, bias)
            .map_err(|e| crate::ops::Error(e.to_string()))
    }
    fn try_linear(
        &mut self,
        selection: (FusionBackend, &Path),
        stream: &mut Stream,
        ops: &Ops,
        x: &Tensor,
        w: &Weight,
        bias: Option<hrx::View<'_>>,
    ) -> Result<Option<Tensor>> {
        let (backend, checkpoint) = selection;
        let shape = Shape { m: x.rows(), k: x.cols(), n: w.shape[0], bias: bias.is_some() };
        let op_error = |e: crate::ops::Error| Error::Message(e.to_string());
        let capture = std::env::var_os("KREA2_FUSION_CAPTURE").map(std::path::PathBuf::from);
        if let Some(root) = capture.filter(|_| !self.captured.contains(&shape.name())) {
            let directory = root.join(shape.name());
            fs::create_dir_all(&directory)?;
            let weights = read(stream, w.values().map_err(op_error)?)?;
            let biases = match bias {
                Some(view) => read(stream, view)?,
                None => vec![],
            };
            let input = read(stream, x.binding().map_err(op_error)?)?;
            if self.checkpoint.is_none() {
                self.checkpoint = Some(hrx::bundle::file_digest(checkpoint)?);
            }
            let checkpoint_digest = self.checkpoint.as_ref().unwrap().clone();
            let case = Case {
                shape: shape.clone(),
                checkpoint: checkpoint_digest,
                weights: hrx::bundle::digest(&weights),
                bias: hrx::bundle::digest(&biases),
                input: hrx::bundle::digest(&input),
            };
            for (name, bytes) in
                [("weights.bin", weights), ("bias.bin", biases), ("input.bin", input)]
            {
                fs::write(directory.join(name), bytes)?;
            }
            q::save(&directory.join("case.json"), &case)?;
            self.captured.insert(shape.name());
        }
        if backend == FusionBackend::Gpu {
            self.reason = "GPU: explicitly selected".into();
            return Ok(None);
        }
        if let Some(index) = self.entries.iter().position(|(key, _, _)| key == &shape) {
            let entry = self.entries.remove(index).unwrap();
            self.entries.push_back(entry);
        } else {
            if self.entries.len() == 2 {
                self.entries.pop_front();
            }
            let loaded = (|| -> Result<Pilot> {
                let path = q::directory()?.join(format!("{}.json", shape.name()));
                let bytes = fs::read(path).map_err(|error| {
                    if error.kind() == std::io::ErrorKind::NotFound {
                        Error::Message("no saved NPU qualification for this shape".into())
                    } else {
                        error.into()
                    }
                })?;
                let record: Record = serde_json::from_slice(&bytes)?;
                if backend == FusionBackend::Auto && !record.qualified() {
                    return Err(Error::Unsupported(
                        "no passing latency and quality qualification".into(),
                    ));
                }
                let required =
                    shape.storage_bytes().filter(|v| *v <= STORAGE_LIMIT).ok_or_else(|| {
                        Error::Unsupported("fusion storage budget exceeded".into())
                    })?;
                while self
                    .entries
                    .iter()
                    .filter_map(|(_, entry, _)| entry.as_ref())
                    .map(|p| p.bytes)
                    .sum::<usize>()
                    + required
                    > STORAGE_LIMIT
                {
                    self.entries.pop_front();
                }
                let weights = read(stream, w.values().map_err(op_error)?)?;
                let biases = match bias {
                    Some(view) => read(stream, view)?,
                    None => vec![],
                };
                if self.checkpoint.is_none() {
                    self.checkpoint = Some(hrx::bundle::file_digest(checkpoint)?);
                }
                record.verify(
                    &shape,
                    self.checkpoint.as_deref().unwrap(),
                    &hrx::bundle::digest(&weights),
                    &hrx::bundle::digest(&biases),
                )?;
                // The profile directory is the user's trusted, local compiler
                // output store; verify binds its artifacts to this implementation.
                unsafe { Pilot::load(&record, &weights, &biases) }
            })();
            let (pilot, reason) = match loaded {
                Ok(pilot) => (Some(pilot), "NPU: verified fusion specialization".into()),
                Err(error) if backend == FusionBackend::Auto && can_fallback(&error) => {
                    (None, format!("GPU: {error}"))
                }
                Err(error) => return Err(error),
            };
            self.entries.push_back((shape.clone(), pilot, reason));
        }
        let (_, pilot, reason) = self.entries.back().unwrap();
        self.reason = reason.clone();
        let Some(pilot) = pilot else { return Ok(None) };
        let output = ops.tensor(stream, shape.m, shape.n).map_err(op_error)?;
        // After submission, every failure propagates; no fallback reads poisoned data.
        pilot.execute(
            stream,
            x.binding().map_err(op_error)?,
            output.binding().map_err(op_error)?,
        )?;
        Ok(Some(output))
    }
}

fn can_fallback(error: &Error) -> bool {
    match error {
        Error::Unsupported(_)
        | Error::Message(_)
        | Error::Io(_)
        | Error::Json(_)
        | Error::Library(_) => true,
        Error::Context { source, .. } => can_fallback(source),
        _ => false,
    }
}
