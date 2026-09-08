"""Resident-buffer fp16 attention A/B with a CPU oracle and alternating pairs.

--prepare-only compiles kernels and writes inputs without accessing the GPU.
Normal runs require an idle GPU and build/attention-bench (scripts/build.sh).
The fixed baseline revision keeps the 2x target stable as the working tree changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from kernel_test import ROOT, compile_kernel

BASE_STEM = "attention_gqa_lds_f16_wmma"
HEADS, KV, D = 48, 12, 128


def prepare_inputs(directory, tokens, capacity, scale, seed):
    random = np.random.default_rng(seed)
    arrays = []
    for name, heads in (("q", HEADS), ("k", KV), ("v", KV)):
        values = np.zeros((capacity, heads, D), np.float16)
        values[:tokens] = random.standard_normal((tokens, heads, D), dtype=np.float32) * (
            scale if name != "v" else 1.)
        values.tofile(directory / f"{name}.bin")
        arrays.append(values)
    q, k, v = arrays
    rows = np.arange(tokens, dtype=np.uint32) if tokens <= 128 else np.unique(np.concatenate([
        np.linspace(0, tokens - 1, 32, dtype=np.uint32),
        np.array([0, 1, 14, 15, 16, 17, 31, 32, 63, 64], dtype=np.uint32),
        np.arange(tokens - 17, tokens, dtype=np.uint32),
    ]))
    want = np.empty((len(rows), HEADS, D), np.float32)
    # Work one GQA group at a time. All oracle arithmetic is CPU float64;
    # only sampled queries need logits, so memory remains linear in tokens.
    for head in range(KV):
        queries = q[rows, head * 4:head * 4 + 4].astype(np.float64).reshape(-1, D)
        logits = queries @ k[:tokens, head].astype(np.float64).T / np.sqrt(D)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        result = weights @ v[:tokens, head].astype(np.float64)
        want[:, head * 4:head * 4 + 4] = result.reshape(len(rows), 4, D)
    rows.tofile(directory / "rows.bin")
    want.tofile(directory / "want.bin")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", default="attention_query_f16")
    parser.add_argument("--candidate-source", type=Path, help="source path for a candidate stored outside experiments/")
    parser.add_argument("--baseline", default="6405667", help="baseline Git revision")
    parser.add_argument("--tokens", default="4115", help="comma-separated token counts")
    parser.add_argument("--qtiles", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--scale", type=float, default=1., help="Q/K standard deviation")
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--transpose-v", action="store_true", help="include V transpose in every timed candidate call")
    parser.add_argument("--transpose-source", type=Path, help="alternative V packing source with the sage_transpose ABI")
    parser.add_argument("--prepare-only", action="store_true", help="CPU only: no GPU initialization or launch")
    parser.add_argument("--output", type=Path, default=ROOT / "build/attention-2x/paired")
    args = parser.parse_args()
    if args.transpose_source and not args.transpose_v:
        parser.error("--transpose-source requires --transpose-v")
    tokens_list = [int(t) for t in args.tokens.split(",")]
    if any(t < 16 or t > 65536 for t in tokens_list):
        parser.error("tokens must be in 16..65536")
    if not 10 <= args.rounds <= 10000 or not 0 < args.scale <= 4:
        parser.error("rounds must be in 10..10000 and scale in (0, 4]")
    candidate = args.candidate_source
    if candidate is None:
        candidate = ROOT / "experiments" / (args.candidate + ".loom")
        if not candidate.exists():
            candidate = ROOT / "kernels" / (args.candidate + ".loom")
    candidate_source = candidate.read_bytes()
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:kernels/{BASE_STEM}.loom"], cwd=ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    for tokens in tokens_list:
        directory = (args.output / f"{args.candidate}-{tokens}-s{args.scale}-seed{args.seed}").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # A new preparation can replace the source/configuration at this path.
        # Never leave a previous timing report attached to the new manifest,
        # including when compilation, correctness, or the idle gate fails.
        (directory / "result.json").unlink(missing_ok=True)
        capacity = (tokens + 16 + 63) // 64 * 64
        for label, stem, source in (("baseline", BASE_STEM, baseline_source),
                                    ("candidate", args.candidate, candidate_source)):
            source_file = directory / f"{label}.loom"
            source_file.write_bytes(source)
            config = dict(q_stride=HEADS * D, kv_stride=KV * D, out_stride=HEADS * D,
                          tokens=tokens, token_capacity=capacity, scale=D ** -.5)
            compile_kernel(source_file, "krea2_" + stem,
                           {f"krea2.{stem}.{k}": v for k, v in config.items()},
                           directory / f"{label}.hsaco")
        row_count = prepare_inputs(directory, tokens, capacity, args.scale, args.seed)
        command = [str(ROOT / "build/attention-bench"),
                   str(directory / "baseline.hsaco"), "krea2_" + BASE_STEM,
                   str(directory / "candidate.hsaco"), "krea2_" + args.candidate,
                   str(tokens), str(capacity), str(args.qtiles), str(args.rounds),
                   str(row_count), str(directory)]
        if args.transpose_v:
            transpose = directory / "transpose.hsaco"
            transpose_source = args.transpose_source or ROOT / "kernels/sage_transpose.loom"
            (directory / "transpose.loom").write_bytes(transpose_source.read_bytes())
            compile_kernel(directory / "transpose.loom", "krea2_sage_transpose",
                           {"krea2.sage_transpose.width": KV * D,
                            "krea2.sage_transpose.row_capacity": capacity}, transpose)
            command.append(str(transpose))
        manifest = dict(tokens=tokens, capacity=capacity, scale=args.scale, seed=args.seed,
                        qtiles=args.qtiles, transpose_v=args.transpose_v, baseline_revision=args.baseline,
                        baseline_sha256=hashlib.sha256(baseline_source).hexdigest(),
                        candidate_sha256=hashlib.sha256(candidate_source).hexdigest(),
                        command=command, oracle_rows=row_count)
        # Sources alone cannot identify a run when experimenting with compiler
        # lowering or changing the host runner. Preserve the actual binaries too.
        manifest["binary_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ["baseline.hsaco", "candidate.hsaco"] +
            (["transpose.hsaco"] if args.transpose_v else [])
        }
        runner = Path(command[0])
        if runner.exists():
            manifest["runner_sha256"] = hashlib.sha256(runner.read_bytes()).hexdigest()
        if args.transpose_v:
            manifest["transpose_sha256"] = hashlib.sha256((directory / "transpose.loom").read_bytes()).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {directory} ({row_count} CPU oracle rows)", flush=True)
        if args.prepare_only:
            continue
        from bench_i4_gemm import idle_check
        idle_check()
        result = subprocess.run(command, text=True, capture_output=True)
        print(result.stderr, end="")
        print(result.stdout, end="")
        result.check_returncode()
        (directory / "result.json").write_text(
            json.dumps({"manifest": manifest, **json.loads(result.stdout)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
