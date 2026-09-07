"""prepare_{norm,gated,plain}_i4 vs the reference's own quantisation (krea2_ref):
the int4 codes must match exactly except at rounding ties, and the scales to f32."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "reference"))
from kernel_test import compile_kernel, launch, workdir
import krea2_ref as R


def unpack(q: np.ndarray, bits: int = 4) -> np.ndarray:
    if bits == 8:
        return q.view(np.int8)
    lo = (q & 0xF).astype(np.int8); hi = ((q >> 4) & 0xF).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo); hi = np.where(hi > 7, hi - 16, hi)
    return np.stack([lo, hi], axis=-1).reshape(q.shape[0], -1)


def check(name, tmp, tokens, width, x_expected, args, cfg, pad=0, bits=4):
    """x_expected: the f32 row before rotation; compare codes and scales. pad: extra output
    row pitch in elements (the GEMM's k_stride padding); the pad bytes must stay untouched.
    bits 8 checks the int8 twin against the same rotation at 127 levels."""
    h = R.hadamard(256)
    xr = R.rotate_groups(torch.from_numpy(x_expected), h)
    qs, ss = R.quant_rows(xr, 7 if bits == 4 else 127)
    sym, ns = f"krea2_prepare_{name}_i{bits}", f"krea2.prepare_{name}_i{bits}"
    hs = tmp / f"{name}-{bits}.hsaco"
    compile_kernel(ROOT / f"kernels/prepare_{name}_i{bits}.loom", sym, {f"{ns}.width": width, f"{ns}.out_stride": width + pad, **cfg}, hs)
    row_bytes, data_bytes = (width + pad) * bits // 8, width * bits // 8
    sentinel = np.full((tokens, row_bytes), 0xA5, dtype=np.uint8)
    (q, s), t = launch(hs, sym, (tokens, 1, 1), (256, 1, 1), [("i32", tokens)] + args +
                       [("inout_u8", (sentinel, sentinel.shape)), ("out", ((tokens,), np.float32))], tmp, repeat=3)
    assert np.all(q[:, data_bytes:] == 0xA5), "the prepare kernel wrote into the row padding"
    q = q[:, :data_bytes]
    got = unpack(q, bits).astype(np.int64); want = qs.numpy().astype(np.int64)
    mism = (got != want); off_by_one = (np.abs(got - want) == 1)
    bad = mism & ~off_by_one
    scale_err = np.abs(s - ss.numpy().ravel()).max() / np.abs(ss.numpy()).max()
    # the SwiGLU variant keeps its 16384-wide row as f16 in LDS: a few more ties and a
    # scale rounded at f16 precision, both far inside int4's step
    # (at 127 levels the f16 intermediate's ~5e-4 relative error moves about 1.2% of codes by one)
    tol_codes, tol_scale = (5e-3 if bits == 4 else 2e-2, 2e-3) if name == "plain" else (2e-3, 1e-5)
    ok = bad.sum() == 0 and mism.mean() < tol_codes and scale_err < tol_scale
    print(f"  {'PASS' if ok else 'FAIL'} prepare_{name}_i{bits}: tokens={tokens} width={width}  {t['per_launch_us'] / 1e3:.3f} ms  "
          f"codes differ {mism.mean() * 100:.3f}% (all by 1, ties) scale rel err {scale_err:.1e}")
    return ok


def main() -> int:
    ok = True
    rng = np.random.default_rng(0)
    tokens, width = 300, 6144
    with workdir() as tmp:
        tmp = Path(tmp)
        # norm: h f16, norm_scale, mod scale/shift f32
        h = (rng.standard_normal((tokens, width)) * 1.5).astype(np.float16)
        ns = (rng.standard_normal(width) * 0.1).astype(np.float32)
        ms = (rng.standard_normal(width) * 0.2).astype(np.float32)
        sh = (rng.standard_normal(width) * 0.2).astype(np.float32)
        hf = h.astype(np.float32)
        normed = hf / np.sqrt((hf * hf).mean(axis=1, keepdims=True) + 1e-5) * (1 + ns)
        x = (1 + ms) * normed + sh
        normed_mod = x
        ok &= check("norm", tmp, tokens, width, x.astype(np.float32),
                    [("in_f16", h), ("in", ns), ("in", ms), ("in", sh)], {"krea2.prepare_norm_i4.eps": 1e-5})
        # gated: attn f16 [tokens][width], gate f16 [tokens][gate_stride] (a slice of the fused output)
        gate_stride = 15360
        attn = (rng.standard_normal((tokens, width)) * 0.5).astype(np.float16)
        fused = (rng.standard_normal((tokens, gate_stride)) * 0.5).astype(np.float16)
        g = fused[:, :width].astype(np.float32)
        x = attn.astype(np.float32) / (1 + np.exp(-g))
        gated = x
        ok &= check("gated", tmp, tokens, width, x.astype(np.float32),
                    [("in_f16", attn), ("in_f16", fused)], {"krea2.prepare_gated_i4.gate_stride": gate_stride})
        # plain: the fused GEMM's silu(g)*u output [tokens][inter] as is
        inter = 16384
        x = (rng.standard_normal((tokens, inter)) * 0.5).astype(np.float16)
        ok &= check("plain", tmp, tokens, inter, x.astype(np.float32), [("in_f16", x)], {})
        ok &= check("plain", tmp, tokens, inter, x.astype(np.float32), [("in_f16", x)], {}, pad=128)
        # the int8 twins: the same rows at 127 levels, with and without a padded pitch
        ok &= check("norm", tmp, tokens, width, normed_mod.astype(np.float32),
                    [("in_f16", h), ("in", ns), ("in", ms), ("in", sh)], {"krea2.prepare_norm_i8.eps": 1e-5}, bits=8)
        ok &= check("gated", tmp, tokens, width, gated.astype(np.float32),
                    [("in_f16", attn), ("in_f16", fused)], {"krea2.prepare_gated_i8.gate_stride": gate_stride}, bits=8)
        ok &= check("plain", tmp, tokens, inter, x.astype(np.float32), [("in_f16", x)], {}, bits=8)
        ok &= check("plain", tmp, tokens, inter, x.astype(np.float32), [("in_f16", x)], {}, pad=64, bits=8)
        # Signed Hadamard basis rows exercise the largest intermediates; constants
        # previously overflowed even though their normalized rotation fits in f16.
        h256 = R.hadamard(256).numpy()
        x = np.stack([np.full(inter, 5000), np.tile(h256[0] * 16 * 65504, inter // 256),
                      np.full(inter, -65504), np.zeros(inter)]).astype(np.float16)
        ok &= check("plain", tmp, len(x), inter, x.astype(np.float32), [("in_f16", x)], {})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
