"""Accuracy probe for INT4 QK attention using real Krea block inputs.

This historical probe emulates 16-token Q smoothing and quantization in Torch.
The native SA2 backend uses 64-token smoothing; its kernel tests and benchmarks
live in tests/test_sage_attention.py and tools/bench_sage_attention.py.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "reference")]
import krea2_ref as R
from loom_ref import LoomBlocksRef
from krea2_loom import Krea2Blocks


def quantize(x):
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 7
    return (x / scale).round().clamp(-7, 7) * scale


def qkv(weights, x, mods, cos, sin, layer):
    ref = LoomBlocksRef(weights, layers=28)
    p = f"blocks.{layer}"
    scale, shift = mods[layer, :2].cuda()
    z = R.rms_norm(x.cuda().float(), ref.t(p + ".prenorm.scale", torch.float32))
    z = z * (1 + scale) + shift
    result = []
    for name, heads in (("q", 48), ("k", 12), ("v", 12)):
        y = ref.project(f"{p}.attn.w{name}.weight", z).half().view(1, -1, heads, 128)
        if name != "v":
            y = R.rms_norm(y.float(), ref.t(f"{p}.attn.qknorm.{name}norm.scale", torch.float32))
            y = R.apply_rope(y, cos.cuda(), sin.cuda()).half()
        if heads == 12:
            y = y.repeat_interleave(4, dim=2)
        result.append(y[0].transpose(0, 1).float())
    return result


def probe(q, k, v, queries):
    n = q.shape[1]
    selected = torch.linspace(0, n - 1, min(queries, n), device=q.device).long().unique()
    # Means over actual query tokens in 16-row blocks, including partial tails.
    qmean = torch.cat([block.mean(1, keepdim=True).expand_as(block)
                       for block in q.split(16, dim=1)], dim=1)
    kc = k - k.mean(1, keepdim=True)
    qc = q - qmean
    # An orthogonal 128-channel transform applied AFTER rotary positions.
    h = torch.ones(1, 1, device=q.device)
    for _ in range(7):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h /= math.sqrt(128)
    candidates = {
        "raw_per_token_i4": (quantize(q[:, selected]), quantize(k), None),
        "K_centered_per_token_i4": (quantize(q[:, selected]), quantize(kc), None),
        "QK_centered_per_token_i4": (quantize(qc[:, selected]), quantize(kc), qmean[:, selected] @ kc.transpose(1, 2)),
        "K_centered_hadamard_per_token_i4": (quantize(q[:, selected] @ h), quantize(kc @ h), None),
    }
    target = ((q[:, selected] @ k.transpose(1, 2)) / math.sqrt(128)).softmax(-1) @ v
    results = {}
    for name, (a, b, correction) in candidates.items():
        scores = a @ b.transpose(1, 2)
        if correction is not None:
            scores += correction
        out = (scores / math.sqrt(128)).softmax(-1) @ v
        cs = torch.nn.functional.cosine_similarity(out.flatten(), target.flatten(), dim=0)
        per_head = torch.nn.functional.cosine_similarity(out.flatten(1), target.flatten(1), dim=1)
        results[name] = dict(cosine=float(cs), worst_head_cosine=float(per_head.min()),
                             relative_l2=float((out - target).norm() / target.norm()))
    return results


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=ROOT / "build/fixture_step0.pt")
    ap.add_argument("--model", type=Path, default=Path.home() / "krea2-models/krea2_turbo_bf16.safetensors")
    ap.add_argument("--layers", default="0,13,27", help="zero-based sampled block indices")
    ap.add_argument("--queries", type=int, default=128)
    args = ap.parse_args()
    layers = sorted(set(map(int, args.layers.split(","))))
    if not layers or layers[0] < 0 or layers[-1] >= 28 or args.queries < 1:
        ap.error("layers must be 0..27 and queries positive")
    fx = torch.load(args.fixture, map_location="cpu")
    state = fx["x"].half()
    print(json.dumps({"tokens": state.shape[0], "sampled_queries": min(args.queries, state.shape[0]),
                      "timestep": float(fx["timestep"].item()), "layers": layers}), flush=True)
    session = Krea2Blocks(state.shape[0], layers=28)
    try:
        previous = 0
        for layer in layers:
            if layer > previous:
                state = session.forward(state, fx["mods"], fx["cos"], fx["sin"],
                                        first_block=previous, block_count=layer - previous)
            p = f"blocks.{layer}"
            names = [p + ".prenorm.scale", p + ".attn.qknorm.qnorm.scale", p + ".attn.qknorm.knorm.scale"]
            names += [p + f".attn.w{c}.weight" for c in "qkv"]
            with safe_open(args.model, framework="pt", device="cpu") as source:
                weights = {name: source.get_tensor(name) for name in names}
            q, k, v = qkv(weights, state, fx["mods"], fx["cos"], fx["sin"], layer)
            print(json.dumps({"layer": layer, "results": probe(q, k, v, args.queries)}), flush=True)
            del q, k, v, weights
            previous = layer
    finally:
        session.close()


if __name__ == "__main__":
    main()
