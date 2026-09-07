"""Export the Krea 2 transformer blocks for the Loom runtime: W4A4 or W8A8 ConvRot layouts.

    python3 tools/export_weights.py [--layers 28] [--out build/weights]
    python3 tools/export_weights.py --bits 8 --source ~/comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors --out build/weights_int8

--bits 8 takes ComfyUI's int8 ConvRot checkpoint verbatim: its rows are already rotated by
the same group-256 Hadamard the kernels use (checked on the first block against the bf16
release when it is present), with one f32 scale per output row; nothing is requantised.

Per block, seven GEMM weights [N][K] bf16 become int4 nibbles (low first) after the
group-256 Hadamard rotation along K (comfy-quants' regular Hadamard, H4 Kronecker
power, 1/16), quantised symmetrically per output row (scale = absmax / 7), and fused
the way the runtime launches them:
  qkvg : rows [ wq (6144) | wk (1536) | wv (1536) | gate (6144) ]   K = 6144  -> N = 15360
  wo   : [6144]                                                     K = 6144
  gu   : rows [ gate o..o+15 | up o..o+15 ] x 1024 (interleaved)      K = 6144  -> N = 32768
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


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """[2*inter][K] with rows in 16-row groups [gate o..o+15 | up o..o+15], so the fused
    GEMM's waves hold the gate and up fragments of the same outputs side by side."""
    inter, k = gate.shape
    return torch.stack([gate.view(inter // 16, 16, k), up.view(inter // 16, 16, k)], dim=1).reshape(2 * inter, k)


def quantize(w: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N][K] on the GPU -> (packed int4 [N][K/2] u8, scale [N] f32), rotated along K."""
    wr = R.rotate_groups(w.float(), h)
    q, s = R.quant_int4_rows(wr)
    return pack_i4(q).cpu(), s.reshape(-1).float().cpu()


def int8_rows(w: dict, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """ComfyUI's int8 ConvRot rows and per-row f32 scale, verbatim."""
    q = w[f"{name}.weight"]
    if q.dtype != torch.int8:
        raise SystemExit(f"{name}.weight is {q.dtype}, not int8: pass the int8 ConvRot checkpoint with --bits 8")
    return q, w[f"{name}.weight_scale"].float().reshape(-1)


def check_rotation(w: dict, name: str, h: torch.Tensor, reference: Path) -> None:
    """The int8 rows must be the bf16 rows rotated by our Hadamard: compare the first block."""
    if not reference.is_file():
        print(f"  (no {reference}: rotation convention not checked)", flush=True)
        return
    from safetensors import safe_open
    with safe_open(str(reference), "pt", device=w[f"{name}.weight"].device.type) as f:
        bf16 = f.get_tensor(f"{name}.weight")
    rotated = R.rotate_groups(bf16.float(), h)
    q, s = int8_rows(w, name)
    dequantized = q.float() * s[:, None]
    cos = torch.nn.functional.cosine_similarity(rotated.flatten(), dequantized.flatten(), dim=0).item()
    print(f"  rotation check {name}: cosine {cos:.6f} between our rotation of the bf16 rows and the int8 rows", flush=True)
    if cos < 0.999:
        raise SystemExit("the int8 checkpoint's rotation does not match the kernels' Hadamard")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--out", default=str(ROOT / "build/weights"))
    ap.add_argument("--source", default=str(TURBO))
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4, help="8: ComfyUI's int8 ConvRot rows verbatim (W8A8)")
    ap.add_argument("--reference", default=str(TURBO), help="the bf16 release, for the --bits 8 rotation check")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = "cuda"
    h = R.hadamard(R.HADAMARD_GROUP).to(dev)
    w = load_file(a.source, device=dev)
    blobs: list[tuple[str, torch.Tensor]] = []          # (name, cpu tensor)
    def add(name, t):
        blobs.append((name, t.contiguous().cpu()))
    if a.bits == 8:
        check_rotation(w, "blocks.0.attn.wq", h, Path(a.reference))
    for i in range(a.layers):
        p = f"blocks.{i}"
        if a.bits == 8:
            parts = [int8_rows(w, f"{p}.attn.{n}") for n in ("wq", "wk", "wv", "gate")]
            gate, up = int8_rows(w, f"{p}.mlp.gate"), int8_rows(w, f"{p}.mlp.up")
            fused = {"qkvg": (torch.cat([q for q, _ in parts]), torch.cat([s for _, s in parts])),
                     "wo": int8_rows(w, f"{p}.attn.wo"),
                     "gu": (interleave_gate_up(gate[0], up[0]), interleave_gate_up(gate[1][:, None], up[1][:, None]).reshape(-1)),
                     "down": int8_rows(w, f"{p}.mlp.down")}
            for name, (q, s) in fused.items():
                add(f"{p}.{name}.q", q.contiguous()); add(f"{p}.{name}.s", s.contiguous())
        else:
            qkvg = torch.cat([w[f"{p}.attn.wq.weight"], w[f"{p}.attn.wk.weight"], w[f"{p}.attn.wv.weight"], w[f"{p}.attn.gate.weight"]], dim=0)
            for name, mat in (("qkvg", qkvg), ("wo", w[f"{p}.attn.wo.weight"]),
                              ("gu", interleave_gate_up(w[f"{p}.mlp.gate.weight"], w[f"{p}.mlp.up.weight"])),
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
                                                    head_dim=R.HEAD_DIM, inter=16384, group=R.HADAMARD_GROUP, bits=a.bits), indent=1))
    print(f"wrote {out}/weights.bin: {offset / 1e9:.2f} GB, {len(manifest)} tensors, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
