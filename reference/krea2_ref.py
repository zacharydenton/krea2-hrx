"""Krea 2 transformer, transcribed from diffusers' Krea2Transformer2DModel onto the
ComfyUI checkpoint key names (the format of every Krea 2 file on this box).

This is the oracle for the Loom port: plain torch, no diffusers dependency, every
block exposed. `blocks_forward` is the part the Loom runtime replaces; everything
around it (text fusion, embeddings, final layer) stays here.

Quantization modes for judging W4A4 ConvRot before any kernel exists:
  quant="none"  bf16 linears (weights as stored)
  quant="w4a4"  per-GEMM: Hadamard rotation (group 256, along K) of both operands,
                weights int4 per-output-row symmetric, activations int4 per-token
                symmetric (dynamic), exact integer products, f32 scale -- the
                arithmetic the Loom int4 GEMM implements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

EPS = 1e-5
ROPE_THETA = 1000.0
AXES = (32, 48, 48)
HEADS, KV_HEADS, HEAD_DIM = 48, 12, 128
HIDDEN = HEADS * HEAD_DIM          # 6144
TXT_DIM, TXT_HEADS, TXT_LAYERS = 2560, 20, 12
HADAMARD_GROUP = 256


# ----------------------------------------------------------------------------- pieces
def rms_norm(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Zero-centred RMSNorm in f32: multiplier is 1 + scale."""
    return F.rms_norm(x.float(), (x.shape[-1],), weight=scale.float() + 1.0, eps=EPS).to(x.dtype)


def hadamard(size: int) -> torch.Tensor:
    """comfy-quants' regular Hadamard: Kronecker power of H4, normalised by 1/sqrt(size)."""
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = torch.ones(1, 1)
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    assert h.shape[0] == size, size
    return h / math.sqrt(size)


def rotate_groups(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Apply the group Hadamard along the last axis: x.view(..., K/g, g) @ H^T."""
    g = h.shape[0]
    shape = x.shape
    return (x.reshape(-1, shape[-1] // g, g).float() @ h.T).reshape(shape).to(x.dtype)


def quant_rows(x: torch.Tensor, levels: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quantization of the last axis: q in -levels..levels, scale = absmax / levels
    (7 for int4, 127 for int8)."""
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-30) / float(levels)
    q = torch.round(x.float() / scale).clamp(-levels, levels)
    return q, scale


def quant_int4_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int4 per row of the last axis: q in -7..7, scale = absmax / 7."""
    return quant_rows(x, 7)


@dataclass
class QuantLinear:
    """A linear layer prepared for W4A4 (or W8A8): rotated integer weights and row scales,
    activations rotated and quantised per token to the same number of levels."""
    q: torch.Tensor          # [N, K] values in -levels..levels (stored as int8)
    scale: torch.Tensor      # [N, 1] f32
    h: torch.Tensor          # the Hadamard block
    levels: int = 7

    @classmethod
    def prepare(cls, w: torch.Tensor, h: torch.Tensor) -> "QuantLinear":
        wr = rotate_groups(w.float(), h)
        q, scale = quant_int4_rows(wr)
        return cls(q.to(torch.int8), scale, h)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xr = rotate_groups(x.float(), self.h)
        xq, xs = quant_rows(xr, self.levels)                  # per token
        acc = xq @ self.q.float().T                           # exact in f32 (|sum| < 2^24)
        return (acc * xs * self.scale.T).to(x.dtype)


class Linear:
    """bf16 linear or its W4A4 / W8A8 stand-in, chosen once per layer. int8: ComfyUI's
    already-rotated int8 rows and their per-row scales, activations at 127 levels."""
    def __init__(self, w: torch.Tensor, quant: str, h: torch.Tensor | None, int8: tuple | None = None):
        self.w = w
        if quant == "w8a8":
            q, scale = int8
            self.q = QuantLinear(q.to(torch.int8), scale.float().reshape(-1, 1), h, 127)
        else:
            self.q = QuantLinear.prepare(w, h) if quant == "w4a4" else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.q is not None:
            return self.q(x)
        return F.linear(x, self.w.to(x.dtype))


def rope_tables(position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """diffusers' Krea2RotaryPosEmbed: per axis, cos/sin of pos * theta^(-2i/d) in f64,
    each frequency repeated twice (interleaved), axes concatenated -> [S, 128] f32."""
    cos_out, sin_out = [], []
    for axis, dim in enumerate(AXES):
        pos = position_ids[:, axis].to(torch.float64)
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim))
        angles = torch.outer(pos, freqs)
        cos_out.append(angles.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(angles.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, S, H, D]; interleaved pairs: (x0, x1) -> (x0*c - x1*s, x1*c + x0*s)."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos[None, :, None, :] + x_rot.float() * sin[None, :, None, :]).to(x.dtype)


def position_ids(text_len: int, grid_h: int, grid_w: int, device) -> torch.Tensor:
    ids = torch.zeros(text_len + grid_h * grid_w, 3, device=device)
    img = torch.zeros(grid_h, grid_w, 3, device=device)
    img[..., 1] = torch.arange(grid_h, device=device)[:, None]
    img[..., 2] = torch.arange(grid_w, device=device)[None, :]
    ids[text_len:] = img.reshape(-1, 3)
    return ids


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = (t.float() * 1e3)[:, None, None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


# ----------------------------------------------------------------------------- the model
class Krea2Ref:
    def __init__(self, weights: dict[str, torch.Tensor], quant: str = "none", device="cuda", dtype=torch.bfloat16,
                 layers: int = 28):
        self.w, self.dev, self.dtype, self.quant = weights, device, dtype, quant
        self.layers = layers
        self.h = hadamard(HADAMARD_GROUP).to(device) if quant in ("w4a4", "w8a8") else None
        self._lin = {}

    def t(self, name: str, dtype=None) -> torch.Tensor:
        return self.w[name].to(self.dev, dtype or self.dtype)

    def lin(self, name: str, quantized: bool = True) -> Linear:
        if name not in self._lin:
            quant = self.quant if quantized else "none"
            int8 = (self.w[name], self.w[name.replace(".weight", ".weight_scale")]) if quant == "w8a8" else None
            self._lin[name] = Linear(self.w[name] if quant == "w8a8" else self.t(name), quant, self.h, int8)
        return self._lin[name]

    # --- attention shared by the text fusion and the blocks -----------------------------
    def attention(self, p: str, x: torch.Tensor, heads: int, kv_heads: int, cos=None, sin=None, quantized=True):
        b, s, _ = x.shape
        q = self.lin(f"{p}.wq.weight", quantized)(x).view(b, s, heads, HEAD_DIM)
        k = self.lin(f"{p}.wk.weight", quantized)(x).view(b, s, kv_heads, HEAD_DIM)
        v = self.lin(f"{p}.wv.weight", quantized)(x).view(b, s, kv_heads, HEAD_DIM)
        gate = self.lin(f"{p}.gate.weight", quantized)(x)
        q = rms_norm(q, self.t(f"{p}.qknorm.qnorm.scale", torch.float32))
        k = rms_norm(k, self.t(f"{p}.qknorm.knorm.scale", torch.float32))
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if heads != kv_heads:
            k = k.repeat_interleave(heads // kv_heads, dim=2)
            v = v.repeat_interleave(heads // kv_heads, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        o = o.reshape(b, s, -1) * torch.sigmoid(gate)
        return self.lin(f"{p}.wo.weight", quantized)(o)

    def swiglu(self, p: str, x: torch.Tensor, quantized=True):
        g = self.lin(f"{p}.gate.weight", quantized)(x)
        u = self.lin(f"{p}.up.weight", quantized)(x)
        return self.lin(f"{p}.down.weight", quantized)(F.silu(g) * u)

    # --- text side (bf16 always; runs once per prompt) -----------------------------------
    def text_fusion_block(self, p: str, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(f"{p}.attn", rms_norm(x, self.t(f"{p}.prenorm.scale", torch.float32)), TXT_HEADS, TXT_HEADS, quantized=False)
        x = x + self.swiglu(f"{p}.mlp", rms_norm(x, self.t(f"{p}.postnorm.scale", torch.float32)), quantized=False)
        return x

    def text_in(self, text_states: torch.Tensor) -> torch.Tensor:
        """[B, T, 12, 2560] tapped hidden states -> [B, T, 6144]."""
        b, s, n, d = text_states.shape
        x = text_states.reshape(b * s, n, d).to(self.dtype)
        for i in range(2):
            x = self.text_fusion_block(f"txtfusion.layerwise_blocks.{i}", x)
        x = x.reshape(b, s, n, d).permute(0, 1, 3, 2)
        x = F.linear(x, self.t("txtfusion.projector.weight")).squeeze(-1)
        for i in range(2):
            x = self.text_fusion_block(f"txtfusion.refiner_blocks.{i}", x)
        x = rms_norm(x, self.t("txtmlp.0.scale", torch.float32))
        x = F.linear(x, self.t("txtmlp.1.weight"), self.t("txtmlp.1.bias"))
        return F.linear(gelu_tanh(x), self.t("txtmlp.3.weight"), self.t("txtmlp.3.bias"))

    def image_in(self, packed_latents: torch.Tensor) -> torch.Tensor:
        return F.linear(packed_latents.to(self.dtype), self.t("first.weight"), self.t("first.bias"))

    def time_embed(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (temb [B, 1, 6144], modulation [B, 1, 6, 6144]) before the per-block tables."""
        e = timestep_embedding(timestep).to(self.dtype)
        e = F.linear(gelu_tanh(F.linear(e, self.t("tmlp.0.weight"), self.t("tmlp.0.bias"))), self.t("tmlp.2.weight"), self.t("tmlp.2.bias"))
        mod = F.linear(gelu_tanh(e), self.t("tproj.1.weight"), self.t("tproj.1.bias"))
        return e, mod.unflatten(-1, (6, HIDDEN))

    def block_modulation(self, i: int, mod: torch.Tensor) -> torch.Tensor:
        """[B, 1, 6, 6144] shared modulation + this block's table."""
        return mod + self.t(f"blocks.{i}.mod.lin").view(6, HIDDEN)

    # --- the blocks: what the Loom runtime replaces ---------------------------------------
    def block(self, i: int, x: torch.Tensor, m: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        p = f"blocks.{i}"
        prescale, preshift, pregate, postscale, postshift, postgate = m.unbind(-2)
        a = self.attention(f"{p}.attn", (1.0 + prescale) * rms_norm(x, self.t(f"{p}.prenorm.scale", torch.float32)) + preshift, HEADS, KV_HEADS, cos, sin)
        x = x + pregate * a
        f = self.swiglu(f"{p}.mlp", (1.0 + postscale) * rms_norm(x, self.t(f"{p}.postnorm.scale", torch.float32)) + postshift)
        return x + postgate * f

    def blocks_forward(self, x: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        for i in range(self.layers):
            x = self.block(i, x, self.block_modulation(i, mod), cos, sin)
        return x

    def final(self, x_img: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        m = temb + self.t("last.modulation.lin").view(2, HIDDEN)[None]      # [B, 2, 6144]
        scale, shift = m[:, 0:1], m[:, 1:2]
        x = (1.0 + scale) * rms_norm(x_img, self.t("last.norm.scale", torch.float32)) + shift
        return F.linear(x, self.t("last.linear.weight"), self.t("last.linear.bias"))

    def forward(self, packed_latents: torch.Tensor, text_states: torch.Tensor, timestep: torch.Tensor,
                grid_h: int, grid_w: int) -> torch.Tensor:
        """Velocity for the image tokens, batch 1 layouts: [B, HW, 64], [B, T, 12, 2560], [B]."""
        txt = self.text_in(text_states)
        img = self.image_in(packed_latents)
        x = torch.cat([txt, img], dim=1)
        temb, mod = self.time_embed(timestep)
        cos, sin = rope_tables(position_ids(txt.shape[1], grid_h, grid_w, x.device))
        x = self.blocks_forward(x, mod, cos, sin)
        return self.final(x[:, txt.shape[1]:], temb)


def load_weights(path: str) -> dict[str, torch.Tensor]:
    return load_file(path)
