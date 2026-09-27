#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy>=2",
#   "torch>=2.13,<2.15",
#   "triton-rocm",
#   "safetensors>=0.6",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-rocm"
# url = "https://download.pytorch.org/whl/rocm7.2"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-rocm" }
# triton-rocm = { index = "pytorch-rocm" }
# ///
"""How much accuracy each integer scheme costs, measured without any kernel of ours.

One unquantized forward of the bf16 ComfyUI checkpoint on the quality fixture, then one per
variant, each reporting the relative RMS of its output against the unquantized forward (the
`forward` measurement of quantized_reference.py). Variants fake-quantize the 28 blocks'
linears: dequantized codes through a float matmul equal the integer GEMM up to f32
rounding. `mix-*` variants put W4A4 with group-64 scales on one layer kind and W8A8 on the
rest. See docs/testing.md.

    ./scripts/integer_scheme_study.py --checkpoint krea2_turbo_bf16.safetensors [VARIANT...]
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantized_reference as qr


class StudyLinear:
    """Fake-quantized linear: optional rank-r bf16 branch (SVD of W), then the residual
    rotated by the group Hadamard and quantized per row (or per K-group) to `levels`;
    activations rotated and quantized per token (or per token and K-group)."""

    def __init__(self, w, h, levels, group, rank):
        w = w.float()
        self.h, self.levels, self.group = h, levels, group
        self.l1 = self.l2 = None
        if rank:
            u, s, v = torch.svd_lowrank(w, q=rank + 16, niter=6)
            self.l1 = (u[:, :rank] * s[:rank]).to(torch.bfloat16)      # [N, r]
            self.l2 = v[:, :rank].T.contiguous().to(torch.bfloat16)    # [r, K]
            w = w - self.l1.float() @ self.l2.float()
        wr = qr.rotate_groups(w, h)
        self.q, self.s = self.quant(wr)
        self.q = self.q.to(torch.int8)

    def quant(self, x):
        if self.group:
            shape = x.shape
            xg = x.reshape(*shape[:-1], shape[-1] // self.group, self.group)
            s = xg.abs().amax(-1, keepdim=True).clamp(min=1e-30) / self.levels
            q = torch.round(xg / s).clamp(-self.levels, self.levels)
            return q.reshape(shape), s
        return qr.quant_rows(x, self.levels)

    def deq(self, q, s):
        if self.group:
            shape = q.shape
            return (q.float().reshape(*shape[:-1], shape[-1] // self.group, self.group) * s).reshape(shape)
        return q.float() * s

    def __call__(self, x):
        xr = qr.rotate_groups(x.float(), self.h)
        xq, xs = self.quant(xr)
        y = self.deq(xq, xs) @ self.deq(self.q, self.s).T
        if self.l1 is not None:
            y = y + (x.to(torch.bfloat16) @ self.l2.T @ self.l1.T).float()
        return y.to(x.dtype)


VARIANTS = {
    "w8a8": dict(levels=127, group=None, rank=0),
    "w4a4": dict(levels=7, group=None, rank=0),
    "w4a4-r32": dict(levels=7, group=None, rank=32),
    "w4a4-g64": dict(levels=7, group=64, rank=0),
    "w4a4-g64-r32": dict(levels=7, group=64, rank=32),
}
W8 = dict(levels=127, group=None, rank=0)
G64 = dict(levels=7, group=64, rank=0)
# Mixed: W4A4 g64 on the named layer kinds, W8A8 on the rest.
MIXED = {
    "mix-gateup": (".mlp.gate.", ".mlp.up."),
    "mix-down": (".mlp.down.",),
    "mix-qkvg": (".attn.wq.", ".attn.wk.", ".attn.wv.", ".attn.gate."),
    "mix-wo": (".attn.wo.",),
}


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="*", choices=[*VARIANTS, *MIXED])
    ap.add_argument("--checkpoint", required=True, help="the bf16 ComfyUI-format checkpoint")
    ap.add_argument("--fixture", type=Path, default=root / "build/quality")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="caches none.npy and each output")
    a = ap.parse_args()
    names = a.variants or [*VARIANTS, *MIXED]
    fixture, out_dir = a.fixture, a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((fixture / "job.json").read_text())
    px, steps, mu = meta["size"], meta["steps"], meta["shift"]
    shift = np.float32(np.exp(mu))
    sigma0 = shift / (shift + (np.float32(1) / np.float32(1.0) - np.float32(1)))
    weights = load_file(a.checkpoint, device="cuda")
    text = torch.from_numpy(np.load(fixture / "text.npy")).cuda().bfloat16()[None]
    state = torch.from_numpy(np.load(fixture / "noise.npy")).cuda().bfloat16()[None]
    grid = px // 16
    t = torch.tensor([float(sigma0)], device="cuda", dtype=torch.bfloat16)

    def forward(ref):
        with torch.no_grad():
            return ref.forward(state, text, t, grid, grid).float()

    truth_path = out_dir / "none.npy"
    if truth_path.exists():
        truth = torch.from_numpy(np.load(truth_path)).cuda()
    else:
        began = time.time()
        truth = forward(qr.Krea2Ref(weights, quant="none", device="cuda"))
        np.save(truth_path, truth.cpu().numpy())
        print(f"none: {time.time() - began:.0f} s", flush=True)
    h = qr.hadamard(qr.HADAMARD_GROUP).cuda()
    for name in names:
        cfg = VARIANTS.get(name)
        kinds = MIXED.get(name)
        began = time.time()
        ref = qr.Krea2Ref(weights, quant="none", device="cuda")
        original = ref.lin

        def lin(n, quantized=True, ref=ref, original=original, cfg=cfg, kinds=kinds):
            if not quantized:
                return original(n, quantized)
            if n not in ref._lin:
                c = cfg if kinds is None else (G64 if any(k in n for k in kinds) else W8)
                ref._lin[n] = StudyLinear(ref.t(n), h, **c)
            return ref._lin[n]

        ref.lin = lin
        out = forward(ref)
        rel = ((out - truth).square().mean().sqrt() / truth.square().mean().sqrt()).item()
        np.save(out_dir / f"{name}.npy", out.cpu().numpy())
        print(f"{name}: relative RMS {rel:.5f}  ({time.time() - began:.0f} s)", flush=True)
        del ref, out
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
