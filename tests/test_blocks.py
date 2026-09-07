"""Check every native block against Torch on identical inputs, then check that
composing those blocks exactly reproduces a single native forward. Also report
the full trajectory's quality against the unfused bf16 model. Independent quantized
trajectories amplify rounding at each quantization boundary and are not a reliable
test of an individual kernel's accuracy.

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
from loom_ref import LoomBlocksRef


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", type=int, default=28); ap.add_argument("--fixture", default=str(ROOT / "build/fixture_step0.pt")); ap.add_argument("--profile", action="store_true"); ap.add_argument("--curve", default="", help="comma-separated depths to report, e.g. 1,2,4,8,16,28")
    ap.add_argument("--weights", default=None, help="exported block weights for the native session (default build/weights)")
    ap.add_argument("--int8", default=None, help="ComfyUI int8 ConvRot checkpoint: the reference runs W8A8 on its rows (pair with --weights build/weights_int8)")
    a = ap.parse_args()
    fx = torch.load(a.fixture)
    x, mods, cos, sin = fx["x"], fx["mods"], fx["cos"], fx["sin"]
    tokens = x.shape[0]
    print(f"fixture: {tokens} tokens ({fx['text_len']} text), grid {fx['grid']}, timestep {fx['timestep'].item():.4f}")
    from safetensors import safe_open
    ok = True
    depths = [int(v) for v in a.curve.split(",") if v] or [a.layers]
    # This suite exercises blocks only; avoid loading the text encoder and
    # untested blocks, especially for a one-block regression on the shared GPU.
    def load(path):
        with safe_open(str(path), framework="pt", device="cuda") as checkpoint:
            return {name: checkpoint.get_tensor(name) for name in checkpoint.keys()
                    if name.startswith("blocks.") and int(name.split(".")[1]) < max(depths)}
    w = load(Path.home() / "krea2-models/krea2_turbo_bf16.safetensors")
    ref = LoomBlocksRef(load(a.int8), layers=max(depths), quant="w8a8") if a.int8 else LoomBlocksRef(w, layers=max(depths))
    bf16 = R.Krea2Ref(w, quant="none", device="cuda", dtype=torch.bfloat16, layers=max(depths))
    gc, gs, gm = cos.cuda(), sin.cuda(), mods.cuda()
    native_state, bf_state = x.half(), x[None].cuda().bfloat16()
    composed, bf_outputs = {}, {}
    loom = Krea2Blocks(tokens, layers=max(depths), weights=a.weights)
    try:
        with torch.no_grad():
            for i in range(max(depths)):
                want = ref.block(i, native_state[None].cuda(), gm[i][None, None], gc, gs)[0].float().cpu()
                got = loom.forward(native_state, mods, cos, sin, first_block=i, block_count=1)
                c = cosine(got.float() - native_state.float(), want - native_state.float())
                good = c > 0.99
                ok &= good
                print(f"  {'PASS' if good else 'FAIL'} block {i + 1:2}: same-input update cosine {c:.6f}", flush=True)
                native_state = got
                bf_state = bf16.block(i, bf_state, gm[i][None, None], gc, gs)
                if i + 1 in depths:
                    composed[i + 1] = native_state
                    bf_outputs[i + 1] = bf_state[0].float().cpu()
        for n in depths:
            t0 = time.time()
            got = loom.forward(x, mods, cos, sin, block_count=n)
            dt = time.time() - t0
            exact = torch.equal(got, composed[n])
            c = cosine(got.float() - x.float(), bf_outputs[n] - x.float())
            ok &= exact and c > 0.9
            print(f"  {n:>2} blocks {dt * 1e3:7.0f} ms: composition exact={exact}, update cosine vs bf16 {c:.5f}", flush=True)
            if a.profile and n == depths[-1]:
                loom.profile(True); loom.forward(x, mods, cos, sin, block_count=n); loom.profile(False)
    finally:
        loom.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
