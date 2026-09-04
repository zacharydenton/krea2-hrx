"""rope_qknorm_f16 vs the reference's rms_norm + apply_rope, in place on a fused
[tokens][15360] buffer with q at 0 and k at 6144."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "reference"))
from kernel_test import compile_kernel, launch, report, workdir
import krea2_ref as R

NS, SYM = "krea2.rope_qknorm_f16", "krea2_rope_qknorm_f16"


def main() -> int:
    torch.manual_seed(0)
    tokens, stride, q_heads, kv_heads, d = 200, 15360, 48, 12, 128
    fused = (torch.randn(tokens, stride) * 0.7).half()
    qs = (torch.randn(d) * 0.1).float(); ks = (torch.randn(d) * 0.1).float()
    pos = R.position_ids(8, 12, 16, "cpu")            # 8 text + 192 image tokens
    cos, sin = R.rope_tables(pos)
    q = fused[:, :q_heads * d].view(1, tokens, q_heads, d)
    k = fused[:, 6144:6144 + kv_heads * d].view(1, tokens, kv_heads, d)
    want = fused.clone()
    want[:, :q_heads * d] = R.apply_rope(R.rms_norm(q, qs), cos, sin).reshape(tokens, -1)
    want[:, 6144:6144 + kv_heads * d] = R.apply_rope(R.rms_norm(k, ks), cos, sin).reshape(tokens, -1)
    with workdir() as tmp:
        tmp = Path(tmp); hs = tmp / "rope.hsaco"
        compile_kernel(ROOT / "kernels/rope_qknorm_f16.loom", SYM,
                       {f"{NS}.row_stride": stride, f"{NS}.q_heads": q_heads, f"{NS}.kv_heads": kv_heads,
                        f"{NS}.k_offset": 6144, f"{NS}.eps": 1e-5}, hs)
        outs, t = launch(hs, SYM, (tokens, 1, 1), (256, 1, 1),
                           [("i32", tokens), ("inout_f16", (fused.numpy(), tuple(fused.shape))), ("in", qs.numpy()), ("in", ks.numpy()),
                            ("in", cos.numpy()), ("in", sin.numpy())], tmp, repeat=1)
        out = outs[0]
        ok = report(f"rope_qknorm tokens={tokens}  {t['per_launch_us'] / 1e3:.3f} ms", out, want.float().numpy(), atol=2e-2, rtol=2e-2)
        # v (untouched) must be intact
        ok &= np.array_equal(out[:, 7680:], fused.numpy()[:, 7680:])
        print("  PASS untouched columns intact" if ok else "  FAIL untouched columns changed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
