"""attention_gqa_f16_wmma vs torch SDPA in f32 (GQA expanded), at the Krea 2 layout."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from kernel_test import compile_kernel, launch, report, workdir, ROOT

import os
import subprocess
STEM = os.environ.get("ATTN", "attention_gqa_lds_f16_wmma")
NS, SYM = "krea2." + STEM, "krea2_" + STEM
HEADS, KV, D = 48, 12, 128


def run(tmp: Path, tokens: int, heads=HEADS, kv=KV) -> bool:
    torch.manual_seed(tokens)
    q = (torch.randn(tokens, heads, D) * 0.5).half()
    k = (torch.randn(tokens, kv, D) * 0.5).half()
    v = (torch.randn(tokens, kv, D) * 0.5).half()
    groups = heads // kv
    qf, kf, vf = (t.float().cuda() for t in (q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)))
    want = torch.nn.functional.scaled_dot_product_attention(qf.transpose(0, 1)[None], kf.transpose(0, 1)[None], vf.transpose(0, 1)[None])[0].transpose(0, 1).reshape(tokens, heads * D).cpu().numpy()
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 63) // 64 * 64)
    def pad(t):                              # 16 rows of headroom
        out = np.zeros((capacity, t.shape[1] * D), np.float16); out[:tokens] = t.reshape(tokens, -1).numpy(); return out
    hs = tmp / f"attn_{tokens}.hsaco"
    cfg = {f"{NS}.q_stride": heads * D, f"{NS}.kv_stride": kv * D, f"{NS}.tokens": tokens,
           f"{NS}.token_capacity": capacity, f"{NS}.scale": 1.0 / math.sqrt(D), f"{NS}.out_stride": heads * D}
    if "lds" not in STEM and STEM != "attention_query32":
        cfg[f"{NS}.kv_groups"] = groups
    compile_kernel((ROOT / "kernels" / f"{STEM}.loom") if (ROOT / "kernels" / f"{STEM}.loom").exists() else ROOT / "experiments" / f"{STEM}.loom", SYM, cfg, hs)
    tiles = (tokens + 15) // 16
    if STEM == "attention_query32":
        vt = np.ascontiguousarray(pad(v).T)
        (out,), t = launch(hs, SYM, ((tokens + 31) // 32, kv, 1), (256, 1, 1),
                           [("i32", tokens), ("i32", 0), ("in_f16", pad(q)), ("in_f16", pad(k)), ("in_f16", vt),
                            ("out_f16", ((tokens, heads * D), np.float16))], tmp, repeat=5)
    elif "lds" in STEM:                        # a workgroup of the four query heads per key-value head
        (out,), t = launch(hs, SYM, (tiles, kv, 1), (128, 1, 1),
                           [("i32", tokens), ("i32", 0), ("in_f16", pad(q)), ("in_f16", pad(k)), ("in_f16", pad(v)),
                            ("out_f16", ((tokens, heads * D), np.float16))], tmp, repeat=5)
    else:
        vt = np.ascontiguousarray(pad(v).T)  # [kv_stride][capacity], zero past the sequence
        (out,), t = launch(hs, SYM, (tiles, heads, 1), (32, 1, 1),
                           [("i32", tokens), ("i32", 0), ("in_f16", pad(q)), ("in_f16", pad(k)), ("in_f16", vt),
                            ("out_f16", ((tokens, heads * D), np.float16))], tmp, repeat=5)
    us = t["per_launch_us"]
    flops = 4.0 * tokens * tokens * D * heads
    return report(f"{STEM} tokens={tokens} heads={heads}/{kv}  {us / 1e3:8.3f} ms  {flops / (us * 1e-6) / 1e12:5.1f} TFLOP/s",
                  out, want, atol=2e-2, rtol=2e-2)


def main() -> int:
    global ROOT, STEM, NS, SYM
    ok = True
    with workdir() as tmp:
        for tokens in (100, 4608):
            ok &= run(Path(tmp), tokens)
        STEM = "attention_query32"
        NS, SYM = "krea2." + STEM, "krea2_" + STEM
        for tokens in (17, 32, 65, 2047, 2048, 4115):
            ok &= run(Path(tmp), tokens)
        # Exercise the optional 32-key generator without rewriting shipped kernels.
        project = ROOT
        generated = Path(tmp) / "generated"
        (generated / "tools").mkdir(parents=True)
        (generated / "kernels").mkdir()
        (generated / "experiments").mkdir()
        generator = generated / "tools/gen_attention_lds.py"
        generator.write_text((project / "tools/gen_attention_lds.py").read_text())
        STEM = "attention_test_lds32"
        NS, SYM = "krea2." + STEM, "krea2_" + STEM
        subprocess.run([sys.executable, str(generator)], check=True,
                       env=dict(os.environ, ATTN_TILE="32", ATTN_STEM=STEM, ATTN_HOIST="4", ATTN_QLDS="1"), stdout=subprocess.DEVNULL)
        ROOT = generated
        ok &= run(Path(tmp), 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
