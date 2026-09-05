"""Smoothed INT4 QK kernel: correctness and timings include native preprocessing."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from kernel_test import compile_kernel
from probe_attention_i4 import quantize


def run(q, k, v, directory, stem="attention_sage_i4_fast", repeat=5):
    tokens, heads, d = q.shape
    kv = k.shape[1]
    capacity = max((tokens + 47) // 32 * 32, (tokens + 63) // 64 * 64)
    cfg = {"q_stride": heads * d, "kv_stride": kv * d, "out_stride": heads * d,
           "tokens": tokens, "token_capacity": capacity, "scale": d ** -0.5}
    hsaco = directory / f"{stem}-{tokens}.hsaco"
    if not hsaco.exists():  # Per-test temporary directory; reuse across timing rounds.
        compile_kernel(ROOT / "kernels" / f"{stem}.loom", "krea2_" + stem,
                       {f"krea2.{stem}.{key}": val for key, val in cfg.items()}, hsaco)
    for name, array in (("q", q), ("k", k), ("v", v)):
        padded = np.zeros((capacity, array.shape[1], d), np.float16)
        padded[:tokens] = array.cpu().numpy()
        padded.tofile(directory / f"{name}.bin")
    torch.cuda.synchronize()  # Keep oracle GPU work out of native timings.
    result = subprocess.run([str(ROOT / "build/sage-runner"), str(hsaco), "krea2_" + stem,
                             str(tokens), str(heads), str(kv), str(capacity), str(directory), str(repeat)],
                            check=True, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    out = torch.from_numpy(np.fromfile(directory / "out.bin", np.float16).reshape(tokens, heads, d)).cuda().float()
    return out, json.loads(result.stdout)


def oracle(q, k, v):
    q = q.cuda().float().transpose(0, 1)
    k = k.cuda().float().repeat_interleave(4, 1).transpose(0, 1)
    v = v.cuda().float().repeat_interleave(4, 1).transpose(0, 1)
    km = k.mean(1, keepdim=True)
    qm = torch.cat([b.mean(1, keepdim=True).expand_as(b) for b in q.split(64, 1)], 1)
    kc = k - km
    # Query chunks bound the oracle workspace at the real sequence length.
    outputs = []
    for start in range(0, q.shape[1], 128):
        a, mean = q[:, start:start+128], qm[:, start:start+128]
        scores = quantize(a - mean) @ quantize(kc).transpose(1, 2) + mean @ kc.transpose(1, 2)
        outputs.append((scores / math.sqrt(128)).softmax(-1) @ v)
    return torch.cat(outputs, 1).transpose(0, 1)


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


@torch.no_grad()
def main():
    with tempfile.TemporaryDirectory() as tmp:
        for tokens in (16, 48, 65, 100, 4115):
            torch.manual_seed(tokens)
            q = (torch.randn(tokens, 48, 128) * .5 + torch.randn(1, 48, 128) * 2).half()
            k = (torch.randn(tokens, 12, 128) * .5 + torch.randn(1, 12, 128) * 2).half()
            v = torch.randn(tokens, 12, 128).half()
            want = oracle(q, k, v)
            out, timing = run(q, k, v, Path(tmp))
            cs = cosine(out, want)
            print(json.dumps(dict(tokens=tokens, cosine_vs_i4_oracle=cs, **timing)), flush=True)
            assert torch.isfinite(out).all() and cs > 0.9999
            pp, pt = run(q, k, v, Path(tmp), "attention_sage_i4_fast_prefetch")
            print(json.dumps(dict(tokens=tokens, prefetch_exact=bool(torch.equal(pp, out)), **pt)), flush=True)
            assert torch.equal(pp, out)
        # A zero centered range must still produce a finite, uniform softmax.
        q = torch.full((65, 48, 128), 2., dtype=torch.float16)
        k = torch.full((65, 12, 128), -3., dtype=torch.float16)
        v = torch.randn(65, 12, 128).half()
        want = v.float().mean(0).repeat_interleave(4, 0).expand(65, -1, -1).cuda()
        out, _ = run(q, k, v, Path(tmp))
        pp, _ = run(q, k, v, Path(tmp), "attention_sage_i4_fast_prefetch")
        assert torch.equal(out, pp)
        assert torch.isfinite(out).all() and torch.allclose(out, want, atol=1e-3, rtol=1e-3)
        print("PASS zero-range quantization and partial smoothing group", flush=True)


if __name__ == "__main__":
    main()
