"""Interleaved timing of attention kernels at the Krea 2 shape, best of N rounds.

    python3 tools/ab_attention.py STEM[:qtiles] STEM[:qtiles] ... [--tokens 4115] [--rounds 5]

Two or more kernels, rotated in order each round so drift cancels, each checked against
Torch's attention on the first round. `:qtiles` is the generator's ATTN_QTILES for that
kernel (16-query tiles per workgroup), which sets the grid and workgroup size. Sources come
from kernels/, else experiments/. Times only on an idle GPU (BENCH_IDLE_WAIT to wait longer,
BENCH_FORCE=1 to time a busy box anyway, which is only good for a smoke test).
"""
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_i4_gemm import idle_check
from kernel_test import compile_kernel, launch, report, workdir, ROOT

HEADS, KV, D = 48, 12, 128


def main() -> int:
    stems, tokens, rounds = [], 4115, 5
    args = sys.argv[1:]
    while args:
        arg = args.pop(0)
        if arg == "--tokens":
            tokens = int(args.pop(0))
        elif arg == "--rounds":
            rounds = int(args.pop(0))
        else:
            stem, _, qtiles = arg.partition(":")
            stems.append((stem, int(qtiles or 1)))
    if len(stems) < 2:
        raise SystemExit(__doc__)
    idle_check()
    torch.manual_seed(0)
    q = (torch.randn(tokens, HEADS, D) * 0.5).half()
    k = (torch.randn(tokens, KV, D) * 0.5).half()
    v = (torch.randn(tokens, KV, D) * 0.5).half()
    qf, kf, vf = (t.float().cuda() for t in (q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)))
    want = torch.nn.functional.scaled_dot_product_attention(
        qf.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None]
    )[0].transpose(0, 1).reshape(tokens, HEADS * D).cpu().numpy()
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 63) // 64 * 64)

    def pad(t):
        out = np.zeros((capacity, t.shape[1] * D), np.float16)
        out[:tokens] = t.reshape(tokens, -1).numpy()
        return out

    ok = True
    with workdir() as tmp:
        tmp = Path(tmp)
        built = {}
        for stem, _ in stems:
            ns, sym = "krea2." + stem, "krea2_" + stem
            hs = tmp / f"{stem}.hsaco"
            cfg = {f"{ns}.q_stride": HEADS * D, f"{ns}.kv_stride": KV * D, f"{ns}.tokens": tokens,
                   f"{ns}.token_capacity": capacity, f"{ns}.scale": 1.0 / math.sqrt(D), f"{ns}.out_stride": HEADS * D}
            source = ROOT / "kernels" / f"{stem}.loom"
            compile_kernel(source if source.exists() else ROOT / "experiments" / f"{stem}.loom", sym, cfg, hs)
            built[stem] = (hs, sym)
        best = {stem: 1e9 for stem, _ in stems}
        for r in range(rounds):
            order = stems[r % len(stems):] + stems[:r % len(stems)]
            for stem, qtiles in order:
                hs, sym = built[stem]
                (out,), t = launch(hs, sym, ((tokens + 16 * qtiles - 1) // (16 * qtiles), KV, 1), (128 * qtiles, 1, 1),
                                   [("i32", tokens), ("i32", KV), ("in_f16", pad(q)), ("in_f16", pad(k)),
                                    ("in_f16", pad(v)), ("out_f16", ((tokens, HEADS * D), np.float16))], tmp, repeat=3)
                if r == 0:
                    ok &= report(f"  {stem} correctness", out, want, atol=2e-2, rtol=2e-2)
                best[stem] = min(best[stem], t["per_launch_us"])
        flops = 4.0 * tokens * tokens * D * HEADS
        baseline = best[stems[0][0]]
        print(f"tokens {tokens}, best of {rounds} rotated rounds x 3 launches:")
        for stem, qtiles in stems:
            print(f"  {stem:<28s} qtiles={qtiles}  {best[stem] / 1e3:8.2f} ms  "
                  f"{flops / (best[stem] * 1e-6) / 1e12:5.1f} TFLOP/s  {baseline / best[stem]:.3f}x")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
