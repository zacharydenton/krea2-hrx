"""Compare the wide INT4 down kernel with a Git revision on resident inputs.

Run scripts/build_host.sh first. Order alternates within each pair; output must
match bit for bit before and after timing. Other GPU jobs can distort timings.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

from kernel_test import ROOT, compile_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="db476cf")
    parser.add_argument("--rounds", type=int, default=80)
    args = parser.parse_args()
    if not 10 <= args.rounds <= 10000:
        parser.error("rounds must be in 10..10000")
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:kernels/gemm_i4_resid.loom"],
        cwd=ROOT,
    )
    with tempfile.TemporaryDirectory(prefix="krea2-down-") as td:
        td = Path(td)
        baseline = td / "baseline.loom"
        baseline.write_bytes(source)
        candidate = td / "candidate.hsaco"
        compile_kernel(
            ROOT / "experiments/gemm_down_i4.loom", "krea2_gemm_down_i4", {}, candidate
        )
        for m in (4096, 4115, 8192, 16896):
            tiles = (m + 127) // 128
            group = min((4, 3, 2), key=lambda g: (tiles + g - 1) // g * g)
            output = td / "baseline.hsaco"
            compile_kernel(
                baseline,
                "krea2_gemm_i4_resid",
                {
                    "krea2.gemm_i4_resid.k_size": 16384,
                    "krea2.gemm_i4_resid.n_size": 6144,
                    "krea2.gemm_i4_resid.m_group": group,
                },
                output,
            )
            subprocess.run(
                [
                    str(ROOT / "build/down-bench"),
                    str(output),
                    str(candidate),
                    str(m),
                    str(group),
                    str(args.rounds),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
