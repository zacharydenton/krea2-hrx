"""Our blocks against ComfyUI's own evaluation (tools/comfy_step.py dumps): every dumped block is
run on ComfyUI's input to that block, and the update it produces is compared with ComfyUI's,
image rows and text rows separately; then the whole stack is chained from ComfyUI's block-0
input and compared at the dumped blocks (drift).

    .venv/bin/python tools/compare_comfy.py [--dumps build/comfy_parity] [--weights CKPT]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference"))
import krea2_ref as R
from krea2_loom import Krea2Blocks, DEFAULT_MODEL


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", type=Path, default=ROOT / "build/comfy_parity")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    meta = json.loads((a.dumps / "meta.json").read_text())
    h_in = torch.from_numpy(np.load(a.dumps / "h_in.npy"))[0].cuda()           # [tokens][6144], ComfyUI's bf16 values
    tvec = torch.from_numpy(np.load(a.dumps / "tvec.npy")).cuda()               # [1][36864]
    tokens, text = h_in.shape[0], meta["text_tokens"]
    grid = meta["size"] // 16
    assert tokens == text + grid * grid, (tokens, text, grid)
    # ComfyUI's per-block modulation: vec + mod.lin in bf16 (DoubleSharedModulation)
    from safetensors import safe_open
    weights = a.weights or str(DEFAULT_MODEL)
    with safe_open(weights, "pt", device="cuda") as f:
        mods = torch.stack([(tvec.bfloat16() + f.get_tensor(f"blocks.{i}.mod.lin").bfloat16()).float().reshape(6, 6144) for i in range(28)])
    cos, sin = R.rope_tables(R.position_ids(text, grid, grid, "cuda"))
    blocks = sorted(int(p.stem[4:]) for p in (a.dumps / "blocks").glob("blk_*.npy"))
    truth = {i: torch.from_numpy(np.load(a.dumps / "blocks" / f"blk_{i:02d}.npy"))[0].cuda() for i in blocks}
    loom = Krea2Blocks(tokens, layers=28, weights=weights)
    print(f"{tokens} tokens ({text} text), {len(blocks)} dumped blocks, sigmas {meta['sigmas'][:3]}...")
    # per block on ComfyUI's input: the update's cosine, image rows and text rows
    prev = h_in
    for i in blocks:
        x_in = truth[i - 1] if (i - 1) in truth else (h_in if i == 0 else None)
        if x_in is None:
            continue
        ours = loom.forward(x_in.bfloat16(), mods, cos, sin, first_block=i, block_count=1).float().cuda()
        du, dt = ours - x_in.float(), truth[i].float() - x_in.float()
        print(f"blk_{i:02d} same input: update cosine image {cosine(du[text:], dt[text:]):.6f} text {cosine(du[:text], dt[:text]):.6f}  "
              f"state cosine {cosine(ours, truth[i]):.6f}  rel rms {(du - dt).norm() / dt.norm():.4f}")
    # chained from block 0
    state = h_in.bfloat16()
    start = 0
    for i in blocks:
        state = loom.forward(state, mods, cos, sin, first_block=start, block_count=i + 1 - start).cuda()
        start = i + 1
        dt, du = truth[i].float() - h_in.float(), state.float() - h_in.float()
        print(f"blk_{i:02d} chained: state cosine {cosine(state.float(), truth[i]):.6f}  cumulative update cosine image {cosine(du[text:], dt[text:]):.6f}  "
              f"rel rms {(state.float() - truth[i].float()).norm() / truth[i].float().norm():.4f}")
    loom.close()


if __name__ == "__main__":
    main()
