import os
"""Torch oracle for the native blocks' fusion and storage boundaries.

krea2_ref models the unfused bf16 transformer. Native blocks instead keep fused
normalization, modulation, SwiGLU and residual arithmetic in fp32, with fp16
storage between kernels. Quantization amplifies those rounding differences over
depth, so kernel agreement needs this separate oracle; model quality is still
compared with the bf16 transformer.
"""
import torch

import krea2_ref as R


def sage_attention(q, k, v):
    """Independent Torch oracle for smoothed INT4 QK and the native correction.

    Query chunks bound score storage; probability/PV arithmetic stays float32
    here to measure the fused kernel's fp16 fragment approximation separately.
    """
    levels = 7 if int(os.environ.get("KREA2_ATTN_QK") or 4) == 4 else 127   # the int8-QK twin's codes
    def quantize(x):
        scale = x.abs().amax(-1, keepdim=True) / levels
        return torch.where(scale > 0, (x / scale).round().clamp(-levels, levels) * scale, 0)

    outputs = []
    for query, key, value in zip(q.float(), k.float(), v.float()):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1).repeat_interleave(4, 0)
        centered_k = key - key.mean(1, keepdim=True)
        k4 = quantize(centered_k).repeat_interleave(4, 0).transpose(1, 2)
        kc = centered_k.half().float().repeat_interleave(4, 0).transpose(1, 2)
        chunks = []
        for block in query.split(64, 1):
            mean = block.mean(1, keepdim=True)
            scores = quantize(block - mean) @ k4 + mean.half().float() @ kc
            chunks.append((scores * (128 ** -0.5)).softmax(-1) @ value)
        outputs.append(torch.cat(chunks, 1).transpose(0, 1))
    return torch.stack(outputs)


class LoomBlocksRef(R.Krea2Ref):
    def __init__(self, weights, device="cuda", layers=28, quant="w4a4"):
        # Keep checkpoint weights in their original bf16 storage. All activation
        # conversions below are explicit, independent of this weight dtype.
        # quant "w8a8": the weights are ComfyUI's int8 ConvRot checkpoint (rows + scales).
        super().__init__(weights, quant=quant, device=device, dtype=torch.bfloat16, layers=layers)

    def project(self, name, x, rotated=False):
        linear = self.lin(name).q
        xr = x if rotated else R.rotate_groups(x.float(), self.h)
        codes, scale = R.quant_rows(xr, linear.levels)
        acc = codes @ linear.q.float().T
        return (acc * linear.scale.T) * scale

    @staticmethod
    def rotate_plain(x):
        # Each radix-4 stage averages signed inputs before its fp16 LDS store.
        # Keep the arithmetic order explicit to match the preparation contract.
        shape = x.shape
        for stride in (1, 4, 16, 64):
            parts = x.reshape(*shape[:-1], -1, 4, stride).float().unbind(-2)
            a, b, c, d = parts
            s01, s23, d01, d23 = a + b, c + d, a - b, c - d
            x = (torch.stack([s01 + d23, s01 - d23, d01 + s23, s23 - d01], dim=-2) * 0.25).half().reshape(shape)
        return x.float() * 16

    def block(self, i, x, m, cos, sin):
        p = f"blocks.{i}"
        prescale, preshift, pregate, postscale, postshift, postgate = m.float().unbind(-2)
        # The residual stream is bf16 with ComfyUI's rounding points: the linear's
        # output, the gated product, and the sum (the block stack's kernels do the same).
        x = x.bfloat16()
        normalized = R.rms_norm(x.float(), self.t(f"{p}.prenorm.scale", torch.float32))
        prepared = normalized * (1 + prescale) + preshift
        b, s, _ = x.shape
        q = self.project(f"{p}.attn.wq.weight", prepared).half().view(b, s, R.HEADS, R.HEAD_DIM)
        k = self.project(f"{p}.attn.wk.weight", prepared).half().view(b, s, R.KV_HEADS, R.HEAD_DIM)
        v = self.project(f"{p}.attn.wv.weight", prepared).half().view(b, s, R.KV_HEADS, R.HEAD_DIM)
        gate = self.project(f"{p}.attn.gate.weight", prepared).half()
        q = R.apply_rope(R.rms_norm(q.float(), self.t(f"{p}.attn.qknorm.qnorm.scale", torch.float32)), cos, sin).half()
        k = R.apply_rope(R.rms_norm(k.float(), self.t(f"{p}.attn.qknorm.knorm.scale", torch.float32)), cos, sin).half()
        attn = sage_attention(q, k, v).reshape(b, s, -1).half()
        attended = attn.float() * torch.sigmoid(gate.float())
        x = (x.float() + (pregate * self.project(f"{p}.attn.wo.weight", attended).bfloat16().float()).bfloat16().float()).bfloat16()
        normalized = R.rms_norm(x.float(), self.t(f"{p}.postnorm.scale", torch.float32))
        prepared = normalized * (1 + postscale) + postshift
        g = self.project(f"{p}.mlp.gate.weight", prepared)
        u = self.project(f"{p}.mlp.up.weight", prepared)
        product = (g * torch.sigmoid(g) * u).half()
        down = self.project(f"{p}.mlp.down.weight", self.rotate_plain(product), rotated=True)
        return (x.float() + (postgate * down.bfloat16().float()).bfloat16().float()).bfloat16()
