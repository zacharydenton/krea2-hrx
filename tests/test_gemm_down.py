"""Wide down projection: full output equivalence and an independent INT4 oracle.

Exercise complete raster groups and tails containing one, two and three tiles.
Signed gates, nonzero residuals, zero scales and sampled CPU integer dots catch
layout or epilogue mistakes that a zero-initialized timing harness can miss.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from kernel_test import ROOT, compile_kernel, launch, workdir


def unpack(packed):
    result = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int32)
    result[:, ::2] = (packed & 15).astype(np.int32)
    result[:, 1::2] = (packed >> 4).astype(np.int32)
    return (result ^ 8) - 8


def main():
    n, k = 6144, 16384
    rng = np.random.default_rng(42)
    weights = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.03, n).astype(np.float32)
    scales[0] = 0
    gate = rng.uniform(-1, 1, n).astype(np.float32)
    gate[-1] = 0
    cols = np.array([0, 15, 16, 127, 128, n - 2, n - 1])
    w = unpack(weights[cols])
    with workdir() as td:
        td = Path(td)
        candidate = td / "wide.hsaco"
        compile_kernel(
            ROOT / "experiments/gemm_down_i4.loom", "krea2_gemm_down_i4", {}, candidate
        )
        for m in (4096, 4097, 4107, 4353, 4609, 16896):
            packed = rng.integers(0, 256, (m, k // 2), dtype=np.uint8)
            activation_scale = rng.uniform(0.001, 0.03, m).astype(np.float32)
            activation_scale[-1] = 0
            residual = rng.uniform(-2, 2, (m, n)).astype(np.float16)
            tiles = (m + 127) // 128
            group = min((4, 3, 2), key=lambda g: (tiles + g - 1) // g * g)
            baseline = td / "baseline.hsaco"
            compile_kernel(
                ROOT / "kernels/gemm_i4_resid.loom",
                "krea2_gemm_i4_resid",
                {
                    "krea2.gemm_i4_resid.k_size": k,
                    "krea2.gemm_i4_resid.n_size": n,
                    "krea2.gemm_i4_resid.m_group": group,
                },
                baseline,
            )
            args = [
                ("i32", m),
                ("in_u8", packed),
                ("in_u8", weights),
                ("in", scales),
                ("in", activation_scale),
                ("inout_f16", (residual, (m, n))),
                ("in", gate),
            ]
            reference, _ = launch(
                baseline,
                "krea2_gemm_i4_resid",
                (n // 128, (tiles + group - 1) // group * group, 1),
                (256, 1, 1),
                args,
                td,
            )
            actual, _ = launch(
                candidate,
                "krea2_gemm_down_i4",
                (n // 128, (m + 255) // 256, 1),
                (256, 1, 1),
                args,
                td,
            )
            np.testing.assert_array_equal(
                actual[0].view(np.uint16), reference[0].view(np.uint16)
            )
            rows = np.array([0, 15, 16, 255, 256, 4095, m - 2, m - 1])
            dots = (unpack(packed[rows]) @ w.T).astype(np.float32)
            expected = dots * scales[cols] * activation_scale[rows, None]
            expected = (
                residual[np.ix_(rows, cols)].astype(np.float32) + expected * gate[cols]
            ).astype(np.float16)
            np.testing.assert_array_equal(actual[0][np.ix_(rows, cols)], expected)
            print(
                f"PASS down M={m}: exact full output and sampled integer oracle",
                flush=True,
            )


if __name__ == "__main__":
    main()
