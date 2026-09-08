"""Compare the two tuned gfx1151 attention kernels, including GPU preprocessing.

Synthetic inputs, warmed native kernels; not a full-image latency benchmark.
Requires scripts/build.sh. The native inference libraries do not use Torch.

--against DIR pairs this tree against another checkout of the repo (a git worktree
with its own scripts/build.sh done): each round runs both trees' preprocessing
and kernel on the same inputs, alternating order, and reports the paired ratio.
Time on an idle GPU only (the check refuses a busy box unless BENCH_FORCE is set).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
from test_sage_attention import run
from bench_i4_gemm import idle_check


def run_in_tree(tree, stem, tokens, repeat, temp):
    """Run another checkout's harness (its build/ binaries, its kernels) on the inputs in temp."""
    code = (
        "import sys, json, numpy as np, torch\n"
        f"sys.path.insert(0, {str(Path(tree) / 'tests')!r})\n"
        "from pathlib import Path\n"
        "import test_sage_attention as t\n"
        f"d = Path({str(temp)!r})\n"
        f"q = torch.from_numpy(np.fromfile(d / 'q.src', np.float16).reshape({tokens}, 48, 128))\n"
        f"k = torch.from_numpy(np.fromfile(d / 'k.src', np.float16).reshape({tokens}, 12, 128))\n"
        f"v = torch.from_numpy(np.fromfile(d / 'v.src', np.float16).reshape({tokens}, 12, 128))\n"
        f"out, times = t.run(q, k, v, d / 'other', {stem!r}, {repeat})\n"
        "out.cpu().numpy().astype(np.float16).tofile(d / 'other.out')\n"
        "print(json.dumps(times))\n"
    )
    env = dict(os.environ, KREA2_TEST_BIN=str(Path(tree) / "build"))
    result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True, env=env, cwd=tree)
    times = json.loads(result.stdout.strip().splitlines()[-1])
    out = torch.from_numpy(np.fromfile(Path(temp) / "other.out", np.float16).reshape(tokens, 48, 128)).cuda().float()
    return out, times


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="4115,8192")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3, help="alternate kernel order between rounds")
    ap.add_argument("--against", type=Path, help="another checkout to pair against (see the module docstring)")
    ap.add_argument("--no-idle-check", action="store_true")
    args = ap.parse_args()
    if not args.no_idle_check:
        idle_check()
    counts = list(map(int, args.tokens.split(",")))
    if any(t < 16 or t > 16896 for t in counts) or args.repeat < 1 or args.rounds < 1:
        ap.error("tokens must be 16..16896 and repeat positive")
    with tempfile.TemporaryDirectory() as temp:
        for tokens in counts:
            torch.manual_seed(tokens)
            q = (torch.randn(tokens, 48, 128) * .5 + torch.randn(1, 48, 128) * 2).half()
            k = (torch.randn(tokens, 12, 128) * .5 + torch.randn(1, 12, 128) * 2).half()
            v = torch.randn(tokens, 12, 128).half()
            reference = None
            stems = ("attention_sage_i4_fast", "attention_sage_i4_fast_prefetch")
            if args.against:
                (Path(temp) / "other").mkdir(exist_ok=True)
                for name, array in (("q", q), ("k", k), ("v", v)):
                    array.numpy().tofile(Path(temp) / f"{name}.src")
                stem = stems[0] if tokens < 8192 else stems[1]
                sides = {"this": lambda: run(q, k, v, Path(temp), stem, args.repeat),
                         "other": lambda: run_in_tree(args.against, stem, tokens, args.repeat, temp)}
                ratios = []
                for round_id in range(args.rounds):
                    order = ("this", "other") if round_id % 2 == 0 else ("other", "this")
                    results = {}
                    for side in order:
                        out, times = sides[side]()
                        if reference is None:
                            reference = out
                        results[side] = dict(times, exact=bool(torch.equal(out, reference)))
                    ratios.append(results["other"]["total_ms"] / results["this"]["total_ms"])
                    print(json.dumps(dict(round=round_id, tokens=tokens, kernel=stem, this=results["this"], other=results["other"],
                                          speedup_total=ratios[-1])), flush=True)
                print(json.dumps(dict(tokens=tokens, kernel=stem, median_speedup_total=sorted(ratios)[len(ratios) // 2])), flush=True)
                continue
            for round_id in range(args.rounds):
                for stem in (stems if round_id % 2 == 0 else stems[::-1]):
                    out, times = run(q, k, v, Path(temp), stem, args.repeat)
                    record = dict(round=round_id, tokens=tokens, kernel=stem, **times)
                    if reference is None:
                        reference = out
                    record["exact"] = bool(torch.equal(out, reference))
                    assert record["exact"]
                    print(json.dumps(record), flush=True)



if __name__ == "__main__":
    main()
