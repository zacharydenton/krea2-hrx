"""Compare the two tuned gfx1151 attention kernels, including GPU preprocessing.

Synthetic inputs, warmed native kernels; not a full-image latency benchmark.
Requires scripts/build_host.sh. The native inference libraries do not use Torch.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_sage_attention import run


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="4115,8192")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3, help="alternate kernel order between rounds")
    args = ap.parse_args()
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
