"""The Loom blocks vs the reference on the fixture (one real denoising step's block
input, modulations and RoPE tables). Two comparisons: against the reference in the
same W4A4 arithmetic (agreement to f16 storage), and against bf16 (the quantisation
error itself, which the image-level check judges).

    python3 tests/test_blocks.py [--layers N]"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference"))
import krea2_ref as R
from krea2_loom import Krea2Blocks


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=28); ap.add_argument("--fixture", default=str(ROOT / "build/fixture_step0.pt")); ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    fx = torch.load(a.fixture)
    x, mods, cos, sin = fx["x"], fx["mods"], fx["cos"], fx["sin"]
    tokens = x.shape[0]
    print(f"fixture: {tokens} tokens ({fx['text_len']} text), grid {fx['grid']}, timestep {fx['timestep'].item():.4f}")
    from safetensors.torch import load_file
    w = load_file(str(Path.home() / "krea2-models/krea2_turbo_bf16.safetensors"), device="cuda")
    ok = True
    loom = Krea2Blocks(tokens, layers=a.layers)
    got = loom.forward(x, mods, cos, sin)                       # cold: module loads, first allocations
    t0 = time.time(); got = loom.forward(x, mods, cos, sin); dt = time.time() - t0
    print(f"loom {a.layers} block(s): {dt * 1e3:.0f} ms warm (includes the host copies)")
    if a.profile:
        loom.profile(True); loom.forward(x, mods, cos, sin); loom.profile(False)
    for quant in ("w4a4", "none"):
        ref = R.Krea2Ref(w, quant=quant, device="cuda", dtype=torch.bfloat16, layers=a.layers)
        with torch.no_grad():
            xr = x[None].cuda().to(torch.bfloat16)
            for i in range(a.layers):
                xr = ref.block(i, xr, mods[i][None, None].cuda(), cos.cuda(), sin.cuda())
        want = xr[0].float().cpu()
        c = cosine(got.float(), want); err = (got.float() - want).abs().max().item() / want.abs().max().item()
        # the update the blocks made, which the residual stream's cosine hides
        c_delta = cosine(got.float() - x.float(), want - x.float())
        good = c_delta > (0.99 if quant == "w4a4" else 0.9)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} vs reference {quant:<5s}: stream cosine {c:.6f}  update cosine {c_delta:.5f}  max rel err {err:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
