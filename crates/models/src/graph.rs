//! Text encoder, text fusion, embeddings, projections and VAE decoder.
//! GPU operations use [`krea2_ops`]. Host-side sinusoids, latent unpacking and tile
//! blending preserve the reference's bf16 rounding boundaries.
use std::sync::Arc;

use hrx::{device, Args};
use krea2_checkpoint::Checkpoint;
use krea2_numerics::{from_f32, to_f32};
use krea2_ops::{config, Binary, Config, Norm, Ops, Pool, Scratch, Tensor, Unary};
use krea2_tokenizer::Tokenizer;

use crate::files::Files;
use crate::weights::Weights;
use crate::{Error, Result};

/// The blocks' six modulation vectors, for all 28 of them.
pub const MODULATION_ELEMENTS: usize = 28 * 6 * 6144;

/// The text encoder's hidden width, and the transformer's.
const TEXT_WIDTH: usize = 2560;
const WIDTH: usize = 6144;

/// ComfyUI's three checkpoints, loaded, and the graph over them.
pub struct Models {
    pub ops: Ops,
    pub tokenizer: Tokenizer,
    text: Weights,
    transformer: Weights,
    vae: Weights,
    /// The 28 blocks' modulation tables as one `[28 * 6][6144]` bf16 tensor.
    block_tables: Tensor,
}

impl Models {
    /// `compiler` is the `loom-compile` this graph's auxiliary kernels are
    /// built with; `None` takes `LOOM_COMPILE`, else PATH.
    pub fn open(files: &Files, compiler: Option<&str>) -> Result<Models> {
        Models::load(
            &files.checkpoint,
            &files.text_encoder,
            &files.vae,
            files.tokenizer.as_deref(),
            compiler,
        )
    }

    /// `tokenizer` of `None` uses the copy compiled into `krea2-tokenizer`.
    pub fn load(
        checkpoint: &std::path::Path,
        text_encoder: &std::path::Path,
        vae: &std::path::Path,
        tokenizer: Option<&std::path::Path>,
        compiler: Option<&str>,
    ) -> Result<Models> {
        let pool = Pool::new();
        let models = Models {
            ops: Ops::with_compiler(Arc::clone(&pool), compiler),
            tokenizer: match tokenizer {
                Some(path) => Tokenizer::from_file(path)?,
                None => Tokenizer::embedded()?,
            },
            text: Weights::load(&Checkpoint::open(text_encoder)?, text_name)?,
            transformer: Weights::load(&Checkpoint::open(checkpoint)?, transformer_name)?,
            vae: Weights::load(&Checkpoint::open(vae)?, vae_name)?,
            block_tables: Tensor::new(&pool, 28 * 6, WIDTH)?,
        };
        models.tables()?;
        Ok(models)
    }

    /// `y = x wᵀ + bias`, for a layer named by its prefix.
    fn lin(&self, x: &Tensor, w: &Weights, prefix: &str) -> Result<Tensor> {
        let bias = match w.has(&format!("{prefix}.bias")) {
            true => Some(w.get(&format!("{prefix}.bias"))?.values),
            false => None,
        };
        Ok(self.ops.linear(x, w.get(&format!("{prefix}.weight"))?, bias)?)
    }

    /// Qwen3-VL's 35 layers, returning the 12 layer taps the fusion consumes.
    ///
    /// The prompt's 34-token prefix is context for the encoder and not part of
    /// the conditioning, so the taps start after it.
    pub fn encode(&self, ids: &[i32]) -> Result<Tensor> {
        if !(35..=546).contains(&ids.len()) {
            return Err(Error("text token count must be 35..546".into()));
        }
        let embedding = self.text.get("embed_tokens.weight")?;
        if embedding.shape.len() != 2 || embedding.shape[1] != TEXT_WIDTH {
            return Err(Error("embedding dimensions".into()));
        }
        if ids.iter().any(|&id| id < 0 || id as usize >= embedding.shape[0]) {
            return Err(Error("token id out of range".into()));
        }
        let (count, tokens) = (ids.len(), ids.len() - 34);
        let mut x = self.ops.tensor(count, TEXT_WIDTH)?;
        let taps = self.ops.tensor(tokens * 12, TEXT_WIDTH)?;
        let identifiers = self.ops.pool().scratch(count * 4)?;
        device().write(identifiers.ptr(), ids)?;
        let mut args = Args::new();
        args.i32(x.size() as i32).ptr(embedding.values).ptr(identifiers.ptr()).ptr(x.ptr());
        self.ops.launch(
            "embedding",
            config(&[("wsize", embedding.count), ("rows", count), ("cols", TEXT_WIDTH)]),
            &args,
            x.size().div_ceil(256),
            1,
            256,
        )?;

        for index in 0..35 {
            let layer = format!("layers.{index}");
            let attn = format!("{layer}.self_attn");
            let normed = self.ops.norm(
                &x,
                self.text.get(&format!("{layer}.input_layernorm.weight"))?,
                Norm::Scale,
                1e-6,
            )?;
            let q = self.lin(&normed, &self.text, &format!("{attn}.q_proj"))?;
            let k = self.lin(&normed, &self.text, &format!("{attn}.k_proj"))?;
            let v = self.lin(&normed, &self.text, &format!("{attn}.v_proj"))?;
            let q = self
                .ops
                .norm(
                    &q.view(count * 32, 128, 0)?,
                    self.text.get(&format!("{attn}.q_norm.weight"))?,
                    Norm::Scale,
                    1e-6,
                )?
                .view(count, 4096, 0)?;
            let k = self
                .ops
                .norm(
                    &k.view(count * 8, 128, 0)?,
                    self.text.get(&format!("{attn}.k_norm.weight"))?,
                    Norm::Scale,
                    1e-6,
                )?
                .view(count, 1024, 0)?;
            let q = self.ops.rope(&q, count, 32, 5e6)?;
            let k = self.ops.rope(&k, count, 8, 5e6)?;
            let attended = self.ops.attention(&q, &k, &v, 1, count, 32, 8, 128, true)?;
            let projected = self.lin(&attended, &self.text, &format!("{attn}.o_proj"))?;
            x = self.ops.binary(&x, &projected, Binary::Add)?;

            let normed = self.ops.norm(
                &x,
                self.text.get(&format!("{layer}.post_attention_layernorm.weight"))?,
                Norm::Scale,
                1e-6,
            )?;
            let gate = self.ops.unary(
                &self.lin(&normed, &self.text, &format!("{layer}.mlp.gate_proj"))?,
                Unary::Silu,
            )?;
            let up = self.lin(&normed, &self.text, &format!("{layer}.mlp.up_proj"))?;
            let mixed = self.ops.binary(&gate, &up, Binary::Mul)?;
            let down = self.lin(&mixed, &self.text, &format!("{layer}.mlp.down_proj"))?;
            x = self.ops.binary(&x, &down, Binary::Add)?;

            // Every third layer from the second: twelve taps over 35 layers.
            if index % 3 == 1 {
                let mut args = Args::new();
                args.i32((tokens * TEXT_WIDTH) as i32).ptr(x.ptr()).ptr(taps.ptr());
                self.ops.launch(
                    "tap",
                    config(&[
                        ("xsize", x.size()),
                        ("ysize", taps.size()),
                        ("tap1", (index - 1) / 3 + 1),
                    ]),
                    &args,
                    (tokens * TEXT_WIDTH).div_ceil(256),
                    1,
                    256,
                )?;
            }
        }
        Ok(taps)
    }

    /// One prenorm/attention/postnorm/MLP block of the text fusion tower.
    fn fusion_block(
        &self,
        x: &Tensor,
        prefix: &str,
        batch: usize,
        tokens: usize,
    ) -> Result<Tensor> {
        let w = &self.transformer;
        let normed = self.ops.norm(
            x,
            w.get(&format!("{prefix}.prenorm.scale"))?,
            Norm::OnePlusScale,
            1e-5,
        )?;
        let q = self.lin(&normed, w, &format!("{prefix}.attn.wq"))?;
        let k = self.lin(&normed, w, &format!("{prefix}.attn.wk"))?;
        let v = self.lin(&normed, w, &format!("{prefix}.attn.wv"))?;
        let q = self
            .ops
            .norm(
                &q.view(q.rows() * 20, 128, 0)?,
                w.get(&format!("{prefix}.attn.qknorm.qnorm.scale"))?,
                Norm::OnePlusScale,
                1e-5,
            )?
            .view(x.rows(), TEXT_WIDTH, 0)?;
        let k = self
            .ops
            .norm(
                &k.view(k.rows() * 20, 128, 0)?,
                w.get(&format!("{prefix}.attn.qknorm.knorm.scale"))?,
                Norm::OnePlusScale,
                1e-5,
            )?
            .view(x.rows(), TEXT_WIDTH, 0)?;
        let attended = self.ops.attention(&q, &k, &v, batch, tokens, 20, 20, 128, false)?;
        let gate = self
            .ops
            .unary(&self.lin(&normed, w, &format!("{prefix}.attn.gate"))?, Unary::Sigmoid)?;
        let attended = self.ops.binary(&attended, &gate, Binary::Mul)?;
        let y = self.ops.binary(
            x,
            &self.lin(&attended, w, &format!("{prefix}.attn.wo"))?,
            Binary::Add,
        )?;

        let normed = self.ops.norm(
            &y,
            w.get(&format!("{prefix}.postnorm.scale"))?,
            Norm::OnePlusScale,
            1e-5,
        )?;
        let gate = self
            .ops
            .unary(&self.lin(&normed, w, &format!("{prefix}.mlp.gate"))?, Unary::Silu)?;
        let up = self.lin(&normed, w, &format!("{prefix}.mlp.up"))?;
        let mixed = self.ops.binary(&gate, &up, Binary::Mul)?;
        Ok(self.ops.binary(
            &y,
            &self.lin(&mixed, w, &format!("{prefix}.mlp.down"))?,
            Binary::Add,
        )?)
    }

    /// The 12 layer taps into one conditioning sequence.
    ///
    /// Two blocks mix the taps of each token against one another, a learned
    /// 12-way average projects them to one, and two more mix across tokens.
    pub fn text_fusion(&self, taps: &Tensor) -> Result<Tensor> {
        let projector = self.transformer.get("txtfusion.projector.weight")?;
        if !taps.rows().is_multiple_of(12) || taps.cols() != TEXT_WIDTH || projector.count != 12
        {
            return Err(Error("text fusion dimensions".into()));
        }
        let tokens = taps.rows() / 12;
        let mut x = taps.clone();
        for index in 0..2 {
            x = self.fusion_block(
                &x,
                &format!("txtfusion.layerwise_blocks.{index}"),
                tokens,
                12,
            )?;
        }
        let projected = self.ops.tensor(tokens, TEXT_WIDTH)?;
        let mut args = Args::new();
        args.i32(projected.size() as i32)
            .ptr(x.ptr())
            .ptr(projector.values)
            .ptr(projected.ptr());
        self.ops.launch(
            "fuse",
            config(&[("xsize", x.size()), ("wsize", 12)]),
            &args,
            projected.size().div_ceil(256),
            1,
            256,
        )?;
        x = projected;
        for index in 0..2 {
            x =
                self.fusion_block(&x, &format!("txtfusion.refiner_blocks.{index}"), 1, tokens)?;
        }
        let x = self.ops.norm(
            &x,
            self.transformer.get("txtmlp.0.scale")?,
            Norm::OnePlusScale,
            1e-5,
        )?;
        let x = self.ops.unary(&self.lin(&x, &self.transformer, "txtmlp.1")?, Unary::Gelu)?;
        self.lin(&x, &self.transformer, "txtmlp.3")
    }

    /// The timestep embedding, and the vector the blocks modulate against.
    ///
    /// The sinusoids are worked out on the host at float precision and rounded
    /// to bf16, which is where the reference implementation rounds them.
    pub fn time(&self, timestep: f32) -> Result<(Tensor, Tensor)> {
        let timestep = to_f32(from_f32(timestep));
        let mut values = vec![0u16; 256];
        for index in 0..128 {
            let angle = timestep * 1000.0 * (-(10000f32.ln()) * index as f32 / 128.0).exp();
            values[index] = from_f32(angle.cos());
            values[index + 128] = from_f32(angle.sin());
        }
        let sinusoids = Tensor::from_slice(self.ops.pool(), &values, 1, 256)?;
        let hidden =
            self.ops.unary(&self.lin(&sinusoids, &self.transformer, "tmlp.0")?, Unary::Gelu)?;
        let embedding = self.lin(&hidden, &self.transformer, "tmlp.2")?;
        let projected =
            self.lin(&self.ops.unary(&embedding, Unary::Gelu)?, &self.transformer, "tproj.1")?;
        Ok((embedding, projected))
    }

    /// The latents into the residual stream's width.
    pub fn image_in(&self, latents: &Tensor) -> Result<Tensor> {
        self.lin(latents, &self.transformer, "first")
    }

    /// Every block's modulation table added to this timestep's vector, as the
    /// float32 buffer the block session reads.
    pub fn modulation(&self, vector: &Tensor) -> Result<Scratch> {
        if vector.size() != 6 * WIDTH {
            return Err(Error("modulation dimensions".into()));
        }
        let out = self.ops.pool().scratch(MODULATION_ELEMENTS * 4)?;
        let mut args = Args::new();
        args.i32(MODULATION_ELEMENTS as i32)
            .ptr(vector.ptr())
            .ptr(self.block_tables.ptr())
            .ptr(out.ptr());
        self.ops.launch(
            "modulation",
            config(&[("xsize", vector.size())]),
            &args,
            MODULATION_ELEMENTS.div_ceil(256),
            1,
            256,
        )?;
        Ok(out)
    }

    /// The final norm, its modulation, and the projection back to latents.
    pub fn last(&self, x: &Tensor, embedding: &Tensor) -> Result<Tensor> {
        if embedding.size() != WIDTH || x.cols() != WIDTH {
            return Err(Error("final layer dimensions".into()));
        }
        let table = self.transformer.get("last.modulation.lin")?.tensor(2, WIDTH)?;
        // The scale and the shift share one embedding, so it goes in twice.
        let expanded = self.ops.tensor(2, WIDTH)?;
        device().copy_device_to_device(expanded.ptr(), embedding.ptr(), WIDTH * 2)?;
        device().copy_device_to_device(
            expanded.ptr().offset(WIDTH * 2),
            embedding.ptr(),
            WIDTH * 2,
        )?;
        let modulated = self.ops.binary(&expanded, &table, Binary::Add)?;
        let scale = modulated.view(1, WIDTH, 0)?;
        let shift = modulated.view(1, WIDTH, WIDTH)?;
        let factor = self.ops.tensor(1, WIDTH)?;
        let mut args = Args::new();
        args.i32(WIDTH as i32).ptr(scale.ptr()).ptr(factor.ptr());
        self.ops.launch("unary_one", Config::new(), &args, WIDTH / 256, 1, 256)?;
        let normed = self.ops.norm(
            x,
            self.transformer.get("last.norm.scale")?,
            Norm::OnePlusScale,
            1e-5,
        )?;
        let scaled = self.ops.binary(&normed, &factor, Binary::Mul)?;
        let shifted = self.ops.binary(&scaled, &shift, Binary::Add)?;
        self.lin(&shifted, &self.transformer, "last.linear")
    }

    /// The 28 blocks' modulation tables gathered into one tensor, once.
    fn tables(&self) -> Result<()> {
        for index in 0..28 {
            let table = self.transformer.get(&format!("blocks.{index}.mod.lin"))?;
            if table.count != 6 * WIDTH {
                return Err(Error("block modulation dimensions".into()));
            }
            let destination = self.block_tables.ptr().offset(index * 6 * WIDTH * 2);
            device().copy_device_to_device(destination, table.values, table.count * 2)?;
        }
        Ok(())
    }

    /// One VAE residual block: two normed convolutions plus the skip.
    fn residual(
        &self,
        x: &Tensor,
        prefix: &str,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let skip = match self.vae.has(&format!("{prefix}.conv_shortcut.weight")) {
            true => self.conv(x, height, width, &format!("{prefix}.conv_shortcut"))?,
            false => x.clone(),
        };
        let y = self.ops.norm_silu(x, self.vae.get(&format!("{prefix}.norm1.gamma"))?)?;
        let y = self.conv(&y, height, width, &format!("{prefix}.conv1"))?;
        let y = self.ops.norm_silu(&y, self.vae.get(&format!("{prefix}.norm2.gamma"))?)?;
        let y = self.conv(&y, height, width, &format!("{prefix}.conv2"))?;
        Ok(self.ops.binary(&y, &skip, Binary::Add)?)
    }

    fn conv(&self, x: &Tensor, height: usize, width: usize, prefix: &str) -> Result<Tensor> {
        let bias = self.vae.get(&format!("{prefix}.bias"))?.values;
        let weight = self.vae.get(&format!("{prefix}.weight"))?;
        Ok(self.ops.conv(x, height, width, weight, Some(bias))?)
    }

    /// One tile of latents to RGB, at eight times the resolution.
    fn decode_tile(&self, input: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let (mut h, mut w) = (height, width);
        let x = self.conv(input, h, w, "post_quant_conv")?;
        let x = self.conv(&x, h, w, "decoder.conv_in")?;
        let mut x = self.residual(&x, "decoder.mid_block.resnets.0", h, w)?;

        let normed = self.ops.norm(
            &x,
            self.vae.get("decoder.mid_block.attentions.0.norm.gamma")?,
            Norm::Group,
            1e-5,
        )?;
        let qkv = self.conv(&normed, h, w, "decoder.mid_block.attentions.0.to_qkv")?;
        let d = x.cols();
        let attended = self.ops.attention(
            &self.columns(&qkv, 0, d)?,
            &self.columns(&qkv, d, d)?,
            &self.columns(&qkv, 2 * d, d)?,
            1,
            h * w,
            1,
            1,
            d,
            false,
        )?;
        let projected = self.conv(&attended, h, w, "decoder.mid_block.attentions.0.proj")?;
        x = self.ops.binary(&x, &projected, Binary::Add)?;
        x = self.residual(&x, "decoder.mid_block.resnets.1", h, w)?;

        for block in 0..4 {
            let prefix = format!("decoder.up_blocks.{block}");
            for resnet in 0..3 {
                x = self.residual(&x, &format!("{prefix}.resnets.{resnet}"), h, w)?;
            }
            if block < 3 {
                x = self.ops.upsample(&x, h, w)?;
                h *= 2;
                w *= 2;
                x = self.conv(&x, h, w, &format!("{prefix}.upsamplers.0.resample.1"))?;
            }
        }
        let x = self.ops.norm_silu(&x, self.vae.get("decoder.norm_out.gamma")?)?;
        self.conv(&x, h, w, "decoder.conv_out")
    }

    /// A window of columns, for splitting a fused qkv projection.
    fn columns(&self, x: &Tensor, start: usize, count: usize) -> Result<Tensor> {
        let y = self.ops.tensor(x.rows(), count)?;
        let mut args = Args::new();
        args.i32(y.size() as i32).ptr(x.ptr()).ptr(y.ptr());
        self.ops.launch(
            "columns",
            config(&[
                ("xsize", x.size()),
                ("cols", count),
                ("width", x.cols()),
                ("start1", start + 1),
            ]),
            &args,
            y.size().div_ceil(256),
            1,
            256,
        )?;
        Ok(y)
    }

    /// Packed latents to RGB bytes, decoded in overlapping tiles.
    ///
    /// The VAE's activations are quadratic in tile area, so a large image is
    /// decoded in 32x32-latent tiles with a 64-pixel overlap blended linearly.
    pub fn decode(&self, packed: &Tensor, height: usize, width: usize) -> Result<Vec<u8>> {
        let (h, w) = (height / 8, width / 8);
        let latent = unpack(&packed.download()?, h, w)?;

        // Tiles overlap by 8 latents where there is more than one of them.
        let stride = if h > 32 || w > 32 { 24 } else { 32 };
        let output_stride = stride * 8;
        let mut tiles: Vec<Vec<Tile>> = Vec::new();
        for y in (0..h).step_by(stride) {
            let mut row = Vec::new();
            for x in (0..w).step_by(stride) {
                let (th, tw) = (std::cmp::min(32, h - y), std::cmp::min(32, w - x));
                let mut input = vec![0u16; th * tw * 16];
                for j in 0..th {
                    let source = ((y + j) * w + x) * 16;
                    input[j * tw * 16..(j + 1) * tw * 16]
                        .copy_from_slice(&latent[source..source + tw * 16]);
                }
                let uploaded = Tensor::from_slice(self.ops.pool(), &input, th * tw, 16)?;
                let output = self.decode_tile(&uploaded, th, tw)?;
                row.push(Tile { height: th * 8, width: tw * 8, data: output.download()? });
            }
            tiles.push(row);
        }
        compose(tiles, output_stride, height, width)
    }
}

/// One decoded tile's RGB samples, still in bf16.
struct Tile {
    height: usize,
    width: usize,
    data: Vec<u16>,
}

/// The sampler's 2x2-packed latents into the VAE's `[h * w][16]`, undoing the
/// Wan latent scaling with bf16 rounding at every term.
fn unpack(packed: &[u16], h: usize, w: usize) -> Result<Vec<u16>> {
    const MEAN: [f32; 16] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
        0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ];
    const STDDEV: [f32; 16] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
        1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ];
    if h == 0
        || w == 0
        || !h.is_multiple_of(2)
        || !w.is_multiple_of(2)
        || packed.len() != h * w * 16
    {
        return Err(Error("latent dimensions".into()));
    }
    // The reference divides by a reciprocal it has already rounded, so the
    // rounding happens there and not on the scale itself.
    let inverse: Vec<f32> =
        STDDEV.iter().map(|&s| to_f32(from_f32(1.0 / to_f32(from_f32(s))))).collect();
    let mean: Vec<f32> = MEAN.iter().map(|&m| to_f32(from_f32(m))).collect();
    let mut latent = vec![0u16; h * w * 16];
    for y in 0..h {
        for x in 0..w {
            for channel in 0..16 {
                let source =
                    ((y / 2) * (w / 2) + x / 2) * 64 + channel * 4 + (y % 2) * 2 + x % 2;
                let scaled = to_f32(from_f32(to_f32(packed[source]) / inverse[channel]));
                latent[(y * w + x) * 16 + channel] = from_f32(scaled + mean[channel]);
            }
        }
    }
    Ok(latent)
}

/// The tiles blended into one image, and mapped from [-1, 1] to bytes.
fn compose(
    mut tiles: Vec<Vec<Tile>>,
    stride: usize,
    height: usize,
    width: usize,
) -> Result<Vec<u8>> {
    let mut rgb = vec![0u8; height * width * 3];
    for row in 0..tiles.len() {
        for column in 0..tiles[row].len() {
            if row > 0 {
                let above = &tiles[row - 1][column];
                let blend =
                    [64, above.height, tiles[row][column].height].into_iter().min().unwrap();
                let taken: Vec<u16> = (0..blend * tiles[row][column].width * 3)
                    .map(|i| {
                        let (y, rest) = (
                            i / (tiles[row][column].width * 3),
                            i % (tiles[row][column].width * 3),
                        );
                        above.data[(above.height - blend + y) * above.width * 3 + rest]
                    })
                    .collect();
                blend_into(&mut tiles[row][column], &taken, blend, true);
            }
            if column > 0 {
                let (before, rest) = tiles[row].split_at_mut(column);
                let left = &before[column - 1];
                let tile = &mut rest[0];
                let blend = [64, left.width, tile.width].into_iter().min().unwrap();
                let taken: Vec<u16> = (0..tile.height * blend * 3)
                    .map(|i| {
                        let (y, rest) = (i / (blend * 3), i % (blend * 3));
                        left.data[(y * left.width + left.width - blend) * 3 + rest]
                    })
                    .collect();
                blend_into(tile, &taken, blend, false);
            }
            let tile = &tiles[row][column];
            for y in 0..std::cmp::min(stride, tile.height) {
                if row * stride + y >= height {
                    break;
                }
                for x in 0..std::cmp::min(stride, tile.width) {
                    if column * stride + x >= width {
                        break;
                    }
                    for channel in 0..3 {
                        let sample = to_f32(tile.data[(y * tile.width + x) * 3 + channel]);
                        if !sample.is_finite() {
                            return Err(Error("the VAE produced a nonfinite sample".into()));
                        }
                        let value = (sample * 0.5 + 0.5).clamp(0.0, 1.0);
                        let index =
                            ((row * stride + y) * width + column * stride + x) * 3 + channel;
                        rgb[index] = (value * 255.0).round_ties_even() as u8;
                    }
                }
            }
        }
    }
    Ok(rgb)
}

/// A linear cross-fade over `blend` rows or columns of a tile's edge, rounding
/// each term to bf16 as the reference does.
fn blend_into(tile: &mut Tile, neighbour: &[u16], blend: usize, vertical: bool) {
    let (rows, columns) = if vertical { (blend, tile.width) } else { (tile.height, blend) };
    for y in 0..rows {
        for x in 0..columns {
            for channel in 0..3 {
                let fraction = if vertical { y as f32 } else { x as f32 } / blend as f32;
                let index = (y * tile.width + x) * 3 + channel;
                let theirs = to_f32(neighbour[(y * columns + x) * 3 + channel]);
                let mine = to_f32(tile.data[index]);
                tile.data[index] = from_f32(
                    to_f32(from_f32(theirs * (1.0 - fraction)))
                        + to_f32(from_f32(mine * fraction)),
                );
            }
        }
    }
}

/// The transformer checkpoint: everything but the blocks' quantised linears
/// (those are the block session's), plus each block's modulation table.
fn transformer_name(key: &str) -> String {
    match key.starts_with("blocks.") && !key.ends_with(".mod.lin") {
        true => String::new(),
        false => key.to_string(),
    }
}

/// ComfyUI's Qwen3-VL text encoder: the language model's layers under
/// `model.`, without the vision tower or the LM head.
fn text_name(key: &str) -> String {
    if key.starts_with("model.visual.")
        || key.starts_with("lm_head")
        || key.starts_with("visual.")
    {
        return String::new();
    }
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        return rest.to_string();
    }
    key.strip_prefix("model.").unwrap_or(key).to_string()
}

/// ComfyUI's (Wan-style) VAE names onto diffusers' AutoencoderKLQwenImage
/// names, decoder only: the flat upsamples list is four blocks of three
/// residual blocks and an upsampler, middle is resnet / attention / resnet,
/// head is norm_out / conv_out, conv1 is conv_in and the top-level conv2 is
/// post_quant_conv. Temporal convolutions are not used for a single image.
fn vae_name(key: &str) -> String {
    if key.starts_with("encoder.") || key.starts_with("conv1.") {
        return String::new();
    }
    if let Some(rest) = key.strip_prefix("conv2.") {
        return format!("post_quant_conv.{rest}");
    }
    let Some(rest) = key.strip_prefix("decoder.") else {
        return String::new();
    };
    // "<prefix><index>.<tail>" -> (index, tail)
    let indexed = |prefix: &str| -> Option<(usize, String)> {
        let rest = rest.strip_prefix(prefix)?;
        let (index, tail) = rest.split_once('.')?;
        Some((index.parse().ok()?, tail.to_string()))
    };
    let (base, mut tail) = if let Some((index, tail)) = indexed("upsamples.") {
        let (block, position) = (index / 4, index % 4);
        let base = match position {
            3 => format!("up_blocks.{block}.upsamplers.0"),
            _ => format!("up_blocks.{block}.resnets.{position}"),
        };
        (base, tail)
    } else if let Some((index, tail)) = indexed("middle.") {
        let base = match index {
            0 => "mid_block.resnets.0",
            1 => "mid_block.attentions.0",
            _ => "mid_block.resnets.1",
        };
        (base.to_string(), tail)
    } else if let Some(tail) = rest.strip_prefix("head.0.") {
        ("norm_out".to_string(), tail.to_string())
    } else if let Some(tail) = rest.strip_prefix("head.2.") {
        ("conv_out".to_string(), tail.to_string())
    } else if let Some(tail) = rest.strip_prefix("conv1.") {
        ("conv_in".to_string(), tail.to_string())
    } else {
        // Nothing this decoder runs: the caller skips it rather than failing,
        // which is what an unexpected tensor in a checkpoint deserves.
        return String::new();
    };
    if tail.contains("time_conv") {
        return String::new();
    }
    for (from, to) in [
        ("residual.0.", "norm1."),
        ("residual.2.", "conv1."),
        ("residual.3.", "norm2."),
        ("residual.6.", "conv2."),
        ("shortcut.", "conv_shortcut."),
    ] {
        if let Some(rest) = tail.strip_prefix(from) {
            tail = format!("{to}{rest}");
            break;
        }
    }
    format!("decoder.{base}.{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transformer_keeps_its_modulation_tables_and_drops_the_block_linears() {
        assert_eq!(transformer_name("blocks.7.mod.lin"), "blocks.7.mod.lin");
        assert_eq!(transformer_name("blocks.7.attn.wq.weight"), "");
        assert_eq!(transformer_name("last.linear.weight"), "last.linear.weight");
    }

    #[test]
    fn the_text_encoder_drops_the_vision_tower_and_unwraps_the_language_model() {
        assert_eq!(
            text_name("model.language_model.layers.0.mlp.up_proj.weight"),
            "layers.0.mlp.up_proj.weight"
        );
        assert_eq!(text_name("model.embed_tokens.weight"), "embed_tokens.weight");
        assert_eq!(text_name("model.visual.blocks.0.attn.qkv.weight"), "");
        assert_eq!(text_name("lm_head.weight"), "");
    }

    #[test]
    fn the_vae_names_map_onto_the_diffusers_decoder() {
        // Four blocks of three resnets and an upsampler, flat in the file.
        assert_eq!(
            vae_name("decoder.upsamples.0.residual.2.weight"),
            "decoder.up_blocks.0.resnets.0.conv1.weight"
        );
        assert_eq!(
            vae_name("decoder.upsamples.3.resample.1.weight"),
            "decoder.up_blocks.0.upsamplers.0.resample.1.weight"
        );
        assert_eq!(
            vae_name("decoder.upsamples.4.shortcut.weight"),
            "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"
        );
        assert_eq!(
            vae_name("decoder.middle.1.norm.gamma"),
            "decoder.mid_block.attentions.0.norm.gamma"
        );
        assert_eq!(
            vae_name("decoder.middle.2.residual.6.bias"),
            "decoder.mid_block.resnets.1.conv2.bias"
        );
        assert_eq!(vae_name("decoder.head.2.weight"), "decoder.conv_out.weight");
        assert_eq!(vae_name("conv2.weight"), "post_quant_conv.weight");
        // The encoder, the temporal taps and the input convolution are not
        // part of a single-image decode.
        assert_eq!(vae_name("encoder.conv1.weight"), "");
        assert_eq!(vae_name("conv1.weight"), "");
        assert_eq!(vae_name("decoder.upsamples.0.time_conv.weight"), "");
    }

    #[test]
    fn the_latent_unpack_undoes_the_two_by_two_packing() {
        // One 2x2 patch: 64 packed values become four pixels of 16 channels.
        let packed: Vec<u16> = (0..64).map(|i| from_f32(i as f32 / 64.0)).collect();
        let latent = unpack(&packed, 2, 2).expect("unpacking");
        assert_eq!(latent.len(), 64);
        for y in 0..2 {
            for x in 0..2 {
                for channel in 0..16 {
                    let source = channel * 4 + y * 2 + x;
                    // Every packed value lands exactly once, scaled.
                    let want = to_f32(packed[source]);
                    let got = to_f32(latent[(y * 2 + x) * 16 + channel]);
                    assert!(got.is_finite(), "channel {channel}");
                    assert_ne!((want, got), (0.0, 0.0), "a value went missing");
                }
            }
        }
        assert!(unpack(&packed, 3, 2).is_err(), "an odd latent grid is not packable");
    }
}
