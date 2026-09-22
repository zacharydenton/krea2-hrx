//! Native Loom BF16 projection with FP32 accumulation and GPU handoffs.
use hrx::{
    execution::{
        Access, BindingContract, Buffer, ExecutableGraph, GpuAccess, KernelContract,
        MemoryPlacement, Runtime, RuntimeOptions, Statistics,
    },
    loom::{Compiler, Specialization},
    Constants, Error, Kernel, Result, Stream, View,
};
use serde::{Deserialize, Serialize};

/// Maximum retained activation, weight and partial-sum storage per projection.
pub const STORAGE_LIMIT: usize = 512 << 20;
const TILE_K: usize = 512;

/// Logical BF16 A[M,K] times transposed BF16 weights[N,K].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}
impl Shape {
    fn mt(self) -> usize {
        self.m.div_ceil(8)
    }
    fn chunk_nt(self) -> usize {
        self.n.div_ceil(8).min(256)
    }
    fn nt(self) -> usize {
        self.n.div_ceil(8).div_ceil(self.chunk_nt()) * self.chunk_nt()
    }
    fn kb(self) -> usize {
        self.k.div_ceil(TILE_K)
    }
    fn kp(self) -> usize {
        self.kb() * TILE_K
    }
    // Multi-worker BF16 pipelines currently exhaust Loom stream routing.
    fn lanes(self) -> usize {
        1
    }
    /// Validate dimensions and bound all native allocations before opening hardware.
    pub fn storage_bytes(self) -> Result<usize> {
        if self.m == 0
            || self.m > 4096
            || self.k == 0
            || self.k > 16384
            || self.n == 0
            || self.n > 32768
        {
            return Err(Error::Message(
                "fusion dimensions require M=1..4096, K=1..16384, N=1..32768".into(),
            ));
        }
        let bytes = self.mt() * 8 * self.kp() * 2
            + self.nt() * 8 * self.kp() * 2
            + self.mt() * self.nt() * self.kb() * 64 * 4
            + self.n * 2
            + self.m * self.n * 2
            + 8 * self.kp() * 2
            + self.chunk_nt() * self.kb() * 64 * 4
            + self.chunk_nt() * 8 * self.kp() * 2;
        if bytes > STORAGE_LIMIT {
            return Err(Error::Message(
                "native fusion exceeds its 512 MiB storage limit".into(),
            ));
        }
        Ok(bytes)
    }
}
fn source(template: &str, values: &[(&str, usize)]) -> String {
    // Replace longer keys first to avoid prefix collisions (K versus KB).
    let mut result = template.to_owned();
    let mut values = values.to_vec();
    values.sort_by_key(|(key, _)| std::cmp::Reverse(key.len()));
    for (key, value) in values {
        result = result.replace(&format!("${key}"), &value.to_string());
    }
    result
}
fn contract(bindings: &[(usize, Access)]) -> KernelContract {
    KernelContract {
        bindings: bindings
            .iter()
            .map(|&(bytes, access)| BindingContract {
                bytes,
                access,
                alignment: 64,
                layout: "BF16 packed operands / FP32 partial tiles".into(),
            })
            .collect(),
        constants: vec![],
    }
}

/// Fixed, reusable native specialization. Weights and bias are immutable;
/// every execute reads fresh external activations and replaces the output.
pub struct Projection {
    runtime: Runtime,
    input: Buffer,
    output: Buffer,
    products: ExecutableGraph,
    reduce: ExecutableGraph,
    weights: Buffer,
    partial: Buffer,
    tile_input: Buffer,
    tile_weight: Buffer,
    tile_output: Buffer,
    pack: Kernel,
    shape: Shape,
    bytes: usize,
    /// Digest of the exact generated native source (including the native tile dimensions).
    pub source_digest: String,
    /// Digest of the admitted native XDNA image.
    pub image_digest: String,
}
impl Projection {
    /// Compile with Loom and load the canonical native image. No vendor SDK is used.
    pub fn new(
        shape: Shape,
        weights: &[u16],
        bias: Option<&[u16]>,
        stream: &Stream,
        compiler: Option<&str>,
    ) -> Result<Self> {
        let bytes = shape.storage_bytes()?;
        if weights.len() != shape.n * shape.k || bias.is_some_and(|b| b.len() != shape.n) {
            return Err(Error::Message(
                "fusion weight/bias size differs from its shape".into(),
            ));
        }
        let runtime = Runtime::with_options(RuntimeOptions {
            memory_budget: stream.memory_budget().cloned(),
            ..Default::default()
        })?;
        let npu = runtime.npu(0)?;
        let text = source(
            include_str!("../../kernels/native/fusion.xdna.loom"),
            &[
                ("LANES", shape.lanes()),
                ("NT", shape.chunk_nt() / shape.lanes()),
                ("KB", shape.kb()),
            ],
        );
        let artifact = Compiler::for_target(compiler.map(std::path::Path::new), npu.target())?
            .module(&text)
            .compile(&Specialization::new("fusion"))?;
        let a_bytes = 8 * shape.kp() * 2;
        let b_bytes = shape.chunk_nt() * 8 * shape.kp() * 2;
        let c_bytes = shape.chunk_nt() * shape.kb() * 64 * 4;
        let chunks = shape.nt() / shape.chunk_nt();
        // SAFETY: the checked-in pipeline consumes one padded eight-row block,
        // streams every weight tile, and writes precisely one FP32 partial tile
        // for each output tile / reduction block. Contracts include padded tails.
        let kernel = unsafe {
            npu.load_artifact(
                &artifact,
                shape.lanes() as u16,
                contract(&[
                    (a_bytes, Access::Read),
                    (b_bytes, Access::Read),
                    (c_bytes, Access::Write),
                ]),
            )?
        };
        let input = runtime.allocate(shape.mt() * a_bytes, MemoryPlacement::GpuLocal)?;
        let tile_input = runtime.allocate(a_bytes, MemoryPlacement::Shared(npu.clone()))?;
        let weight_buffer = runtime.allocate(chunks * b_bytes, MemoryPlacement::HostVisible)?;
        {
            let mut packed = weight_buffer.map_write()?;
            for n in 0..shape.n {
                for k in 0..shape.k {
                    let index = (n / 8 * shape.kp() + k) * 8 + n % 8;
                    packed[index * 2..index * 2 + 2]
                        .copy_from_slice(&weights[n * shape.k + k].to_le_bytes());
                }
            }
        }
        let tile_weight = runtime.allocate(b_bytes, MemoryPlacement::Shared(npu.clone()))?;
        let partial =
            runtime.allocate(shape.mt() * chunks * c_bytes, MemoryPlacement::GpuLocal)?;
        let tile_output = runtime.allocate(c_bytes, MemoryPlacement::Shared(npu))?;
        let bias_buffer = runtime.allocate(shape.n * 2, MemoryPlacement::HostVisible)?;
        if let Some(bias) = bias {
            bias_buffer.map_write()?.copy_from_slice(bytemuck::cast_slice(bias));
        }
        let output = runtime.allocate(shape.m * shape.n * 2, MemoryPlacement::GpuLocal)?;
        let gpu_compiler = Compiler::for_stream(compiler.map(std::path::Path::new), stream)?;
        let pack_count = shape.mt() * 8 * shape.kp();
        let pack_source = source(
            include_str!("../../kernels/native/fusion_pack.loom"),
            &[
                ("GRID", pack_count.div_ceil(256)),
                ("M", shape.m),
                ("K", shape.k),
                ("KP", shape.kp()),
                ("COUNT", pack_count),
            ],
        );
        let pack_artifact =
            gpu_compiler.module(&pack_source).compile(&Specialization::new("fusion_pack"))?;
        // SAFETY: the packing kernel checks logical row/column bounds, writes all
        // padded output elements and reads only the declared activation matrix.
        let pack = unsafe { stream.load_artifact(&pack_artifact)? };
        let reduce_source = source(
            include_str!("../../kernels/native/fusion_reduce.loom"),
            &[
                ("GRID", (shape.m * shape.n).div_ceil(256)),
                ("COUNT", shape.m * shape.n),
                ("N", shape.n),
                ("MT", shape.mt()),
                ("NT", shape.nt()),
                ("KB", shape.kb()),
            ],
        );
        let reduce_artifact = gpu_compiler
            .module(&reduce_source)
            .compile(&Specialization::new("fusion_reduce"))?;
        // SAFETY: fixed shapes bound every partial/bias read and output write.
        let reduce = unsafe {
            runtime.load_gpu_kernel(
                reduce_artifact.path(),
                "fusion_reduce",
                [(shape.m * shape.n).div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                contract(&[
                    (partial.len(), Access::Read),
                    (bias_buffer.len(), Access::Read),
                    (output.len(), Access::Write),
                ]),
            )?
        };
        // One native run avoids consuming a hardware context per matrix tile.
        let mut products = runtime.graph();
        products.npu(&kernel, &[tile_input.view(), tile_weight.view(), tile_output.view()])?;
        let mut reduction = runtime.graph();
        reduction.gpu(&reduce, &[partial.view(), bias_buffer.view(), output.view()])?;
        Ok(Self {
            products: products.prepare()?,
            reduce: reduction.prepare()?,
            runtime,
            input,
            output,
            weights: weight_buffer,
            partial,
            tile_input,
            tile_weight,
            tile_output,
            pack,
            shape,
            bytes,
            source_digest: hrx::bundle::digest(text.as_bytes()),
            image_digest: hrx::bundle::file_digest(artifact.path())?,
        })
    }
    /// Complete input packing, NPU products, GPU reduction/bias and result copy.
    pub fn execute(
        &mut self,
        stream: &mut Stream,
        input: View<'_>,
        output: View<'_>,
    ) -> Result<()> {
        if input.len() != self.shape.m * self.shape.k * 2
            || output.len() != self.shape.m * self.shape.n * 2
        {
            return Err(Error::Message(
                "fusion activation extent differs from its shape".into(),
            ));
        }
        // SAFETY: all coordinated input writes are declared, and dispatch stays
        // on this stream. The handoff fences before making bytes visible to NPU.
        unsafe {
            self.runtime.with_gpu_access(
                stream,
                &[GpuAccess { view: self.input.view(), access: Access::Write }],
                |stream, views| {
                    stream.dispatch(
                        &self.pack,
                        [(self.input.len() / 2).div_ceil(256) as u32, 1, 1],
                        [256, 1, 1],
                        &Constants::new(),
                        &[input, views[0]],
                    )
                },
            )?;
        }
        let a_bytes = self.tile_input.len();
        let b_bytes = self.tile_weight.len();
        let c_bytes = self.tile_output.len();
        let chunks = self.shape.nt() / self.shape.chunk_nt();
        for chunk in 0..chunks {
            self.copy(
                stream,
                self.tile_weight.view(),
                self.weights.slice(chunk * b_bytes..(chunk + 1) * b_bytes)?,
            )?;
            for tile in 0..self.shape.mt() {
                self.copy(
                    stream,
                    self.tile_input.view(),
                    self.input.slice(tile * a_bytes..(tile + 1) * a_bytes)?,
                )?;
                self.products.submit()?.wait()?;
                let offset = (tile * chunks + chunk) * c_bytes;
                self.copy(
                    stream,
                    self.partial.slice(offset..offset + c_bytes)?,
                    self.tile_output.view(),
                )?;
            }
        }
        self.reduce.submit()?.wait()?;
        // SAFETY: the declared read is copied on the supplied stream and fenced
        // before releasing the lease. Borrowed external views never escape.
        unsafe {
            self.runtime.with_gpu_access(
                stream,
                &[GpuAccess { view: self.output.view(), access: Access::Read }],
                |stream, views| stream.copy(output, views[0]),
            )
        }
    }
    fn copy(
        &self,
        stream: &mut Stream,
        destination: hrx::execution::BufferView,
        source: hrx::execution::BufferView,
    ) -> Result<()> {
        // SAFETY: both views are declared with their exact access and copied on
        // the leased stream. with_gpu_access fences and updates NPU visibility.
        unsafe {
            self.runtime.with_gpu_access(
                stream,
                &[
                    GpuAccess { view: destination, access: Access::Write },
                    GpuAccess { view: source, access: Access::Read },
                ],
                |stream, views| stream.copy(views[0], views[1]),
            )
        }
    }
    /// Runtime counters for checking that warm replays allocate/import nothing.
    pub fn statistics(&self) -> Statistics {
        self.runtime.statistics()
    }
    /// Retained data allocation size, excluding executable/runtime metadata.
    pub fn storage_bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_shapes_bound_padding_and_workspace() {
        assert!(Shape { m: 0, k: 1, n: 1 }.storage_bytes().is_err());
        assert!(Shape { m: usize::MAX, k: 1, n: 1 }.storage_bytes().is_err());
        assert!(Shape { m: 4096, k: 16384, n: 32768 }.storage_bytes().is_err());
        assert!(Shape { m: 228, k: 2560, n: 6912 }.storage_bytes().unwrap() < STORAGE_LIMIT);
        assert_eq!(Shape { m: 1, k: 1, n: 1 }.storage_bytes().unwrap(), 33284);
    }
}
