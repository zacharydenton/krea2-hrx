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
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=28); ap.add_argument("--fixture", default=str(ROOT / "build/fixture_step0.pt")); ap.add_argument("--profile", action="store_true"); ap.add_argument("--curve", default="", help="comma-separated depths to report, e.g. 1,2,4,8,16,28")
    a = ap.parse_args()
    fx = torch.load(a.fixture)
    x, mods, cos, sin = fx["x"], fx["mods"], fx["cos"], fx["sin"]
    tokens = x.shape[0]
    print(f"fixture: {tokens} tokens ({fx['text_len']} text), grid {fx['grid']}, timestep {fx['timestep'].item():.4f}")
    from safetensors.torch import load_file
    w = load_file(str(Path.home() / "krea2-models/krea2_turbo_bf16.safetensors"), device="cuda")
    ok = True
    depths = [int(v) for v in a.curve.split(",") if v] or [a.layers]
    refs = {q: R.Krea2Ref(w, quant=q, device="cuda", dtype=torch.bfloat16, layers=max(depths)) for q in ("w4a4", "none")}
    # reference outputs after each requested depth, one pass per mode
    ref_out = {q: {} for q in refs}
    for q, ref in refs.items():
        with torch.no_grad():
            xr = x[None].cuda().to(torch.bfloat16)
            for i in range(max(depths)):
                xr = ref.block(i, xr, mods[i][None, None].cuda(), cos.cuda(), sin.cuda())
                if i + 1 in depths:
                    ref_out[q][i + 1] = xr[0].float().cpu()
    for n in depths:
        loom = Krea2Blocks(tokens, layers=n)
        got = loom.forward(x, mods, cos, sin)
        t0 = time.time(); got = loom.forward(x, mods, cos, sin); dt = time.time() - t0
        if a.profile and n == depths[-1]:
            loom.profile(True); loom.forward(x, mods, cos, sin); loom.profile(False)
        loom.close()
        w4, bf = ref_out["w4a4"][n], ref_out["none"][n]
        c_ref = cosine(w4 - x.float(), bf - x.float())
        line = f"  {n:>2} blocks {dt * 1e3:7.0f} ms:"
        for q, want in (("w4a4", w4), ("none", bf)):
            c = cosine(got.float() - x.float(), want - x.float())
            good = c > (0.99 if q == "w4a4" else 0.9)
            ok &= good if n == depths[-1] else True
            line += f"  update cosine vs {q} {c:.5f}"
        print(line + f"  (reference w4a4 vs bf16 {c_ref:.5f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
