"""Compare auxiliary BF16 GEMMs with a Git revision, using resident GPU inputs.

Run scripts/build.sh first. Alternating order reduces clock and scheduling
bias; a busy GPU can still invalidate timings. Output must be bit identical.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

from kernel_test import ROOT, compile_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="2870077")
    parser.add_argument("--rounds", type=int, default=80)
    args = parser.parse_args()
    if not 10 <= args.rounds <= 10000:
        parser.error("rounds must be in 10..10000")
    baseline_name = "gemm_bf16_bf16_nt"
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:kernels/native/{baseline_name}.loom"],
        cwd=ROOT,
    )
    with tempfile.TemporaryDirectory(prefix="krea2-gemm-") as temp:
        temp = Path(temp)
        baseline = temp / "baseline.loom"
        baseline.write_bytes(source)
        for m, n, k in [
            (64, 6144, 2560),
            (4096, 6144, 64),
            (65536, 256, 2304),
            (16384, 512, 4608),
        ]:
            tile = 128 if m >= 128 and n >= 64 and ((m + 63) // 64) % 2 == 0 else 64
            columns = (
                128
                if tile == 128 and n >= 128 and ((n + 63) // 64) % 2 == 0 and k >= 128
                else 64
            )
            name = baseline_name + (
                "_tiled" if columns == 128 else "_wide" if tile == 128 else ""
            )
            outputs = [temp / "baseline.hsaco", temp / "candidate.hsaco"]
            for src, symbol, rows, cols, output in [
                (baseline, baseline_name, 64, 64, outputs[0]),
                (
                    ROOT / "kernels/native" / f"{name}.loom",
                    name,
                    tile,
                    columns,
                    outputs[1],
                ),
            ]:
                config = dict(
                    m=m,
                    n=n,
                    k=k,
                    asize=m * k,
                    bsize=n * k,
                    csize=m * n,
                    astride=m * k,
                    bstride=n * k,
                    grid_x=(n + cols - 1) // cols,
                    grid_y=(m + rows - 1) // rows,
                )
                compile_kernel(
                    src,
                    "krea2_" + symbol,
                    {f"krea2.{symbol}.{key}": val for key, val in config.items()},
                    output,
                )
            subprocess.run(
                [
                    str(ROOT / "build/gemm-bench"),
                    *map(str, outputs),
                    "krea2_" + name,
                    str(m),
                    str(n),
                    str(k),
                    str(tile),
                    str(columns),
                    str(args.rounds),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
