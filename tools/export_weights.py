"""Export the Krea 2 transformer blocks for the Loom runtime: W4A4 ConvRot layouts.

    python3 tools/export_weights.py [--layers 28] [--out build/weights]

Per block, seven GEMM weights [N][K] bf16 become int4 nibbles (low first) after the
group-256 Hadamard rotation along K (comfy-quants' regular Hadamard, H4 Kronecker
power, 1/16), quantised symmetrically per output row (scale = absmax / 7), and fused
the way the runtime launches them:
  qkvg : rows [ wq (6144) | wk (1536) | wv (1536) | gate (6144) ]   K = 6144  -> N = 15360
  wo   : [6144]                                                     K = 6144
  gu   : rows [ mlp.gate (16384) | mlp.up (16384) ]                 K = 6144  -> N = 32768
  down : [6144]                                                     K = 16384
plus f32 vectors: prenorm/postnorm scales, qnorm/knorm scales (128), the per-block
modulation table (6 x 6144). Written as one weights.bin + manifest (name offset bytes)
exactly like the sibling repos, so the C runtime validates every span.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
import krea2_ref as R

TURBO = Path.home() / "krea2-models" / "krea2_turbo_bf16.safetensors"


def pack_i4(q: torch.Tensor) -> torch.Tensor:
    """[N][K] values in -7..7 -> [N][K/2] uint8, low nibble = even k."""
    u = (q.to(torch.int16) & 0xF).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()


def quantize(w: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N][K] on the GPU -> (packed int4 [N][K/2] u8, scale [N] f32), rotated along K."""
    wr = R.rotate_groups(w.float(), h)
    q, s = R.quant_int4_rows(wr)
    return pack_i4(q).cpu(), s.reshape(-1).float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--out", default=str(ROOT / "build/weights"))
    ap.add_argument("--source", default=str(TURBO))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = "cuda"
    h = R.hadamard(R.HADAMARD_GROUP).to(dev)
    w = load_file(a.source, device=dev)
    blobs: list[tuple[str, torch.Tensor]] = []          # (name, cpu tensor)
    def add(name, t):
        blobs.append((name, t.contiguous().cpu()))
    for i in range(a.layers):
        p = f"blocks.{i}"
        qkvg = torch.cat([w[f"{p}.attn.wq.weight"], w[f"{p}.attn.wk.weight"], w[f"{p}.attn.wv.weight"], w[f"{p}.attn.gate.weight"]], dim=0)
        for name, mat in (("qkvg", qkvg), ("wo", w[f"{p}.attn.wo.weight"]),
                          ("gu", torch.cat([w[f"{p}.mlp.gate.weight"], w[f"{p}.mlp.up.weight"]], dim=0)),
                          ("down", w[f"{p}.mlp.down.weight"])):
            q, s = quantize(mat, h)
            add(f"{p}.{name}.q", q); add(f"{p}.{name}.s", s)
        add(f"{p}.prenorm", w[f"{p}.prenorm.scale"].float())
        add(f"{p}.postnorm", w[f"{p}.postnorm.scale"].float())
        add(f"{p}.qnorm", w[f"{p}.attn.qknorm.qnorm.scale"].float())
        add(f"{p}.knorm", w[f"{p}.attn.qknorm.knorm.scale"].float())
        add(f"{p}.mod", w[f"{p}.mod.lin"].float().view(6, -1))
        print(f"  block {i}: {time.time() - t0:.0f} s", flush=True)
    manifest, offset = [], 0
    with open(out / "weights.bin", "wb") as f:
        for name, t in blobs:
            b = t.numpy().tobytes()
            manifest.append(f"{name} {offset} {len(b)} {t.dtype} {'x'.join(map(str, t.shape))}")
            f.write(b); offset += len(b)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "config.json").write_text(json.dumps(dict(layers=a.layers, hidden=R.HIDDEN, heads=R.HEADS, kv_heads=R.KV_HEADS,
                                                    head_dim=R.HEAD_DIM, inter=16384, group=R.HADAMARD_GROUP), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
