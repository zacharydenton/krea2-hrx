"""The three INT4 GEMM epilogues against a float64 oracle, and optionally against a Git revision's kernels.

Every kernel runs at token counts whose raster tails hold one, two and three tiles
of either size, with signed gates, zero scales and a nonzero residual stream. The
float64 oracle catches layout and epilogue mistakes; the reference comparison
(GEMM_I4_BASELINE, a Git revision) must be bit-exact when set, which is how a load-path
or tile change proves it kept the arithmetic.

GEMM_KPAD=128 pads every operand row with random codes past K (kernels with a
k_stride config); the pad must never be read. GEMM_BITS=8 checks the int8 (W8A8) 256-row
kernels instead: int8 rows, 127-level scales, no 128-row twin (GEMM_KPAD then means 64).
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from kernel_test import ROOT, compile_kernel, launch, workdir, bf16_round, bf16_bits

KERNELS = {  # mode -> tile rows -> (source, symbol); the 256-row kernels also take k_stride
    "plain": {128: ("kernels/gemm_i4.loom", "krea2_gemm_i4"), 256: ("kernels/gemm_i4_256.loom", "krea2_gemm_i4_256")},
    "resid": {128: ("kernels/gemm_i4_resid.loom", "krea2_gemm_i4_resid"), 256: ("kernels/gemm_i4_resid_256.loom", "krea2_gemm_i4_resid_256")},
    "swiglu": {128: ("kernels/gemm_i4_swiglu.loom", "krea2_gemm_i4_swiglu"), 256: ("kernels/gemm_i4_swiglu_256.loom", "krea2_gemm_i4_swiglu_256")},
}
K, N = 1152, 384  # small enough for a full float64 oracle; K is not a multiple of 2048
BITS = int(os.environ.get("GEMM_BITS", "4"))
TOKENS = (100, 129, 4096, 4107, 4115, 4353, 4609)


def unpack(packed):
    if BITS == 8:
        return packed.view(np.int8).astype(np.int32)
    result = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int32)
    result[:, ::2] = (packed & 15).astype(np.int32)
    result[:, 1::2] = (packed >> 4).astype(np.int32)
    return (result ^ 8) - 8


def m_group(m, tile):
    tiles = (m + tile - 1) // tile
    if tiles == 1:
        return 1
    return min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))


def oracle(mode, a, w, w_scale, a_scale, gate, residual):
    full = (a.astype(np.float64) @ w.astype(np.float64).T) * w_scale[None, :] * a_scale[:, None]
    if mode == "plain":
        return full
    if mode == "resid":  # ComfyUI's bf16 rounding points on the bf16 stream
        return bf16_round(residual + bf16_round(gate[None, :] * bf16_round(full.astype(np.float32)))).astype(np.float64)
    out = np.empty((full.shape[0], full.shape[1] // 2))
    for group in range(full.shape[1] // 32):
        gate_cols = full[:, group * 32 : group * 32 + 16]
        up_cols = full[:, group * 32 + 16 : group * 32 + 32]
        out[:, group * 16 : group * 16 + 16] = gate_cols / (1 + np.exp(-gate_cols)) * up_cols
    return out


def check(name, actual, expected, atol, rtol):
    actual = actual.astype(np.float64)
    error = np.abs(actual - expected)
    ok = np.all(error <= atol + rtol * np.abs(expected))
    if not ok:
        worst = np.unravel_index(np.argmax(error - rtol * np.abs(expected)), error.shape)
        raise AssertionError(f"{name}: at {worst} got {actual[worst]} want {expected[worst]}")


def run(mode, source, symbol, namespace, m, k, n, kpad, operands, td, tile):
    packed, weights, w_scale, a_scale, gate, residual = operands
    if BITS == 8:
        source, symbol, namespace = Path(str(source).replace("gemm_i4", "gemm_i8")), symbol.replace("gemm_i4", "gemm_i8"), namespace.replace("gemm_i4", "gemm_i8")
    group = 4 if tile == 256 else m_group(m, tile)  # the 256-row kernels shorten their raster tail
    config = {f"{namespace}.k_size": k, f"{namespace}.n_size": n, f"{namespace}.m_group": group,
              f"{namespace}.k_stride": k + kpad}
    hsaco = td / f"{symbol}-{tile}.hsaco"
    compile_kernel(source, symbol, config, hsaco)
    args = [("i32", m), ("in_u8", packed), ("in_u8", weights), ("in", w_scale), ("in", a_scale)]
    if mode == "resid":
        args += [("inout_bf16", (residual, (m, n))), ("in", gate)]
    else:
        args.append(("out_f16", ((m, n // 2 if mode == "swiglu" else n), None)))
    tiles = (m + tile - 1) // tile
    grid = (n // 128, tiles if tile == 256 else (tiles + group - 1) // group * group, 1)
    outputs, _ = launch(hsaco, symbol, grid, (256, 1, 1), args, td)
    return outputs[0]


def as_float(mode, out):
    return (out.astype(np.uint32) << 16).view(np.float32) if mode == "resid" else out


def main():
    baseline_rev = os.environ.get("GEMM_I4_BASELINE", "")  # a Git revision whose 128-row kernels must match bit for bit (opt-in)
    kpad = int(os.environ.get("GEMM_KPAD", "0"))
    modes = os.environ.get("GEMM_I4_MODES", "plain,resid,swiglu").split(",")
    rng = np.random.default_rng(7)
    with workdir() as td:
        td = Path(td)
        for mode in modes:
            source, symbol = KERNELS[mode][128]
            namespace = symbol.replace("krea2_", "krea2.", 1)
            wide_source, wide_symbol = KERNELS[mode][256]
            wide_namespace = wide_symbol.replace("krea2_", "krea2.", 1)
            baseline = None
            if baseline_rev:
                text = subprocess.check_output(["git", "show", f"{baseline_rev}:{source}"], cwd=ROOT)
                baseline = td / f"{mode}-baseline.loom"
                baseline.write_bytes(text)
            n = N
            k = K
            weights = rng.integers(0, 256, (n, (k + kpad) * BITS // 8), dtype=np.uint8)
            w_scale = rng.uniform(0.001, 0.03, n).astype(np.float32) / (1 if BITS == 4 else 16)
            w_scale[3] = 0
            gate = rng.uniform(-1, 1, n).astype(np.float32)
            gate[-1] = 0
            w = unpack(weights[:, : k * BITS // 8])
            for m in TOKENS:
                packed = rng.integers(0, 256, (m, (k + kpad) * BITS // 8), dtype=np.uint8)
                a_scale = rng.uniform(0.001, 0.03, m).astype(np.float32) / (1 if BITS == 4 else 16)
                a_scale[-1] = 0
                residual = bf16_round(rng.uniform(-2, 2, (m, n)).astype(np.float32))
                operands = (packed, weights, w_scale, a_scale, gate, residual)
                expected = oracle(mode, unpack(packed[:, : k * BITS // 8]), w, w_scale, a_scale, gate, residual)
                wide = run(mode, ROOT / wide_source, wide_symbol, wide_namespace, m, k, n, kpad, operands, td, 256)
                # f16 output: half-ulp rounding plus f32 accumulation order
                check(f"{mode} 256-row M={m}", as_float(mode, wide), expected, atol=2e-3 if mode != "resid" else 2e-2, rtol=4e-3 if mode != "resid" else 8e-3)
                if BITS == 8:  # no 128-row int8 kernel
                    print(f"PASS {mode} int8 256-row M={m}: float64 oracle" + (f" (pad {kpad})" if kpad else ""), flush=True)
                    continue
                actual = run(mode, ROOT / source, symbol, namespace, m, k, n, kpad, operands, td, 128)
                check(f"{mode} M={m}", as_float(mode, actual), expected, atol=2e-3 if mode != "resid" else 2e-2, rtol=4e-3 if mode != "resid" else 8e-3)
                np.testing.assert_array_equal(wide.view(np.uint16), actual.view(np.uint16), err_msg=f"{mode} M={m}: 256-row kernel differs from the 128-row kernel")
                if baseline is not None and not kpad:  # older kernels have no k_stride
                    reference = run(mode, baseline, symbol, namespace, m, k, n, 0, (packed[:, : k // 2], weights[:, : k // 2]) + operands[2:], td, 128)
                    np.testing.assert_array_equal(actual.view(np.uint16), reference.view(np.uint16), err_msg=f"{mode} M={m}: differs from {baseline_rev}")
                print(f"PASS {mode} M={m}: float64 oracle, 256-row kernel bit-exact" + (f" (pad {kpad})" if kpad else "") + ("" if baseline is None else f", 128-row bit-exact vs {baseline_rev}"), flush=True)


if __name__ == "__main__":
    main()
