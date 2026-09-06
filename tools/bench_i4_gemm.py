"""Paired A/B of an INT4 GEMM kernel against a Git revision (or another kernel).

Run scripts/build_host.sh first. Both sides run on the same resident operands,
alternating order within each pair, and must agree bit for bit before and after
timing. Use it for a kernel edit (--baseline REV), a different tile
(--candidate-source/--candidate-symbol/--tile-candidate), or the operand-pitch
falsifier (--k-candidate: same kernel, longer K, no output comparison).

Timings only mean something on an idle GPU: no other GPU job, no compile
running (the APU shares one power budget), gpu_busy_percent at 0 beforehand.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from kernel_test import ROOT, compile_kernel

SOURCES = {
    "plain": ("kernels/gemm_i4.loom", "krea2_gemm_i4"),
    "resid": ("kernels/gemm_i4_resid.loom", "krea2_gemm_i4_resid"),
    "swiglu": ("kernels/gemm_i4_swiglu.loom", "krea2_gemm_i4_swiglu"),
}
SHAPES = {  # the block's real (K, N) per epilogue
    "plain": (6144, 15360),
    "resid": (6144, 6144),
    "swiglu": (6144, 32768),
}


def m_group(m, tile):
    tiles = (m + tile - 1) // tile
    if tiles == 1:
        return 1
    return min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))


def idle_check():
    """Wait (BENCH_IDLE_WAIT seconds, default 300) for five consecutive idle GPU readings; refuse a busy box unless BENCH_FORCE is set."""
    paths = list(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
    busy, deadline = [], time.monotonic() + int(os.environ.get("BENCH_IDLE_WAIT", "300"))
    while time.monotonic() < deadline:
        busy = [int(p.read_text()) for p in paths for _ in (0,)]
        time.sleep(0.2)
        if all(int(p.read_text()) == 0 for p in paths):
            streak = 1
            while streak < 5 and all(int(p.read_text()) == 0 for p in paths):
                streak += 1
                time.sleep(0.2)
            if streak == 5:
                busy = []
                break
    load = os.getloadavg()[0]
    if busy or load > 4:
        message = f"box not idle: gpu_busy_percent={busy} load={load:.1f}"
        if os.environ.get("BENCH_FORCE"):
            print("WARNING " + message, flush=True)
        else:
            raise SystemExit(message + " (set BENCH_FORCE=1 to time anyway)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=SOURCES, default="plain")
    parser.add_argument("--baseline", default="HEAD", help="git revision of the baseline kernel source")
    parser.add_argument("--baseline-source", help="baseline kernel path (default: the mode's kernel)")
    parser.add_argument("--baseline-file", type=Path, help="use this file as the baseline instead of a Git revision")
    parser.add_argument("--baseline-symbol")
    parser.add_argument("--candidate-source", help="working-tree kernel path (default: the mode's kernel)")
    parser.add_argument("--candidate-symbol")
    parser.add_argument("--namespace", help="config namespace of the baseline (default from the symbol)")
    parser.add_argument("--candidate-namespace")
    parser.add_argument("--m", default="4115", help="comma-separated token counts")
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k-candidate", type=int, help="falsifier: run the candidate at a different K")
    parser.add_argument("--k-stride", type=int, help="operand row pitch (kernels with a k_stride config)")
    parser.add_argument("--k-stride-candidate", type=int)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--tile-candidate", type=int)
    parser.add_argument("--group", type=int, help="raster group (default: the builders' rule)")
    parser.add_argument("--group-candidate", type=int)
    parser.add_argument("--baseline-shorten", action="store_true", help="the baseline kernel shortens its raster tail (grid rows = tiles)")
    parser.add_argument("--candidate-shorten", action="store_true", help="the candidate kernel shortens its raster tail (grid rows = tiles)")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--json", type=Path, help="append each result line to this file")
    parser.add_argument("--no-idle-check", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.rounds <= 10000:
        parser.error("rounds must be in 10..10000")
    source, symbol = SOURCES[args.mode]
    k, n = SHAPES[args.mode]
    k = args.k or k
    n = args.n or n
    k_cand = args.k_candidate or k
    base_source = args.baseline_source or source
    cand_source = args.candidate_source or source
    base_symbol = args.baseline_symbol or symbol
    cand_symbol = args.candidate_symbol or symbol
    base_ns = args.namespace or base_symbol.replace("krea2_", "krea2.", 1)
    cand_ns = args.candidate_namespace or cand_symbol.replace("krea2_", "krea2.", 1)
    tile_cand = args.tile_candidate or args.tile
    if not args.no_idle_check:
        idle_check()
    if args.baseline_file:
        baseline_text = args.baseline_file.read_bytes()
        args.baseline = str(args.baseline_file)
    else:
        baseline_text = subprocess.check_output(["git", "show", f"{args.baseline}:{base_source}"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="krea2-i4-") as td:
        td = Path(td)
        base_path = td / "baseline.loom"
        base_path.write_bytes(baseline_text)
        for m in (int(x) for x in args.m.split(",")):
            groups = (args.group or (4 if args.baseline_shorten else m_group(m, args.tile)),
                      args.group_candidate or (4 if args.candidate_shorten else m_group(m, tile_cand)))
            grid_groups = (1 if args.baseline_shorten else groups[0], 1 if args.candidate_shorten else groups[1])
            outputs = []
            for side, (src, sym, ns, kk, ks, g) in enumerate((
                (base_path, base_symbol, base_ns, k, args.k_stride, groups[0]),
                (ROOT / cand_source, cand_symbol, cand_ns, k_cand, args.k_stride_candidate, groups[1]),
            )):
                config = {f"{ns}.k_size": kk, f"{ns}.n_size": n, f"{ns}.m_group": g}
                if ks:
                    config[f"{ns}.k_stride"] = ks
                if "config.decl" not in Path(src).read_text():
                    config = {}  # a fixed-shape experiment kernel takes no configs
                out = td / f"side{side}.hsaco"
                compile_kernel(src, sym, config, out)
                outputs.append(out)
            command = [
                str(ROOT / "build/i4-bench"), str(outputs[0]), str(outputs[1]),
                f"mode={args.mode}", f"symbol={base_symbol}", f"symbol_cand={cand_symbol}",
                f"m={m}", f"n={n}", f"k={k}", f"k_cand={k_cand}",
                f"k_stride={args.k_stride or k}", f"k_stride_cand={args.k_stride_candidate or k_cand}",
                f"tile={args.tile}", f"tile_cand={tile_cand}",
                f"group={groups[0]}", f"group_cand={groups[1]}",
                f"grid_group={grid_groups[0]}", f"grid_group_cand={grid_groups[1]}", f"rounds={args.rounds}",
            ]
            line = subprocess.check_output(command, cwd=ROOT, text=True).strip()
            result = json.loads(line)
            result.update(baseline_rev=args.baseline, baseline_source=base_source, candidate_source=cand_source)
            print(json.dumps(result), flush=True)
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
