"""Compile the block kernels for one sequence length into build/kernels/T<tokens>/.

    python3 scripts/build_kernels.py <tokens>

Everything but attention and the V transpose is independent of the sequence length,
but keeping one directory per length keeps the session's contract simple: a session
is created for exactly the tokens its kernels were compiled for."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIDDEN, KV, D, INTER = 6144, 12, 128, 16384
QKVG = HIDDEN + 2 * KV * D + HIDDEN


def compile_one(src: str, root: str, out: Path, cfg: dict) -> None:
    loom = os.environ.get("LOOM_COMPILE") or str(Path.home() / "code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile")
    cmd = [loom, str(ROOT / "kernels" / f"{src}.loom"), "--backend=amdgpu-hal", "--target=gfx1151", f"--root=@{root}", f"--output={out}"]
    cmd += [f"--config={k}={v}" for k, v in cfg.items()]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{src}: {r.stderr.strip()[:600]}")


def build(tokens: int) -> Path:
    capacity = (tokens + 16 + 31) // 32 * 32       # >= tokens + 16 for attention, a multiple of 32 for the transpose
    out = ROOT / "build/kernels" / f"T{tokens}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("prepare_norm_i4", "krea2_prepare_norm_i4", "prepare_norm_i4", {"krea2.prepare_norm_i4.width": HIDDEN, "krea2.prepare_norm_i4.eps": 1e-5}),
        ("prepare_gated_i4", "krea2_prepare_gated_i4", "prepare_gated_i4", {"krea2.prepare_gated_i4.width": HIDDEN, "krea2.prepare_gated_i4.gate_stride": QKVG}),
        ("prepare_swiglu_i4", "krea2_prepare_swiglu_i4", "prepare_swiglu_i4", {"krea2.prepare_swiglu_i4.width": INTER, "krea2.prepare_swiglu_i4.gate_stride": 2 * INTER}),
        ("gemm_i4", "krea2_gemm_i4", "gemm_qkvg", {"krea2.gemm_i4.k_size": HIDDEN, "krea2.gemm_i4.n_size": QKVG}),
        ("gemm_i4", "krea2_gemm_i4", "gemm_gu", {"krea2.gemm_i4.k_size": HIDDEN, "krea2.gemm_i4.n_size": 2 * INTER}),
        ("gemm_i4_resid", "krea2_gemm_i4_resid", "gemm_wo", {"krea2.gemm_i4_resid.k_size": HIDDEN, "krea2.gemm_i4_resid.n_size": HIDDEN}),
        ("gemm_i4_resid", "krea2_gemm_i4_resid", "gemm_down", {"krea2.gemm_i4_resid.k_size": INTER, "krea2.gemm_i4_resid.n_size": HIDDEN}),
        ("rope_qknorm_f16", "krea2_rope_qknorm_f16", "rope_qknorm", {"krea2.rope_qknorm_f16.row_stride": QKVG, "krea2.rope_qknorm_f16.q_heads": 48, "krea2.rope_qknorm_f16.kv_heads": KV, "krea2.rope_qknorm_f16.k_offset": HIDDEN, "krea2.rope_qknorm_f16.eps": 1e-5}),
        ("transpose_f16", "krea2_transpose_f16", "transpose_v", {"krea2.transpose_f16.cols": KV * D, "krea2.transpose_f16.row_stride": QKVG, "krea2.transpose_f16.row_capacity": capacity}),
        ("attention_gqa_f16_wmma", "krea2_attention_gqa_f16_wmma", "attention", {"krea2.attention_gqa_f16_wmma.q_stride": QKVG, "krea2.attention_gqa_f16_wmma.kv_stride": QKVG, "krea2.attention_gqa_f16_wmma.kv_groups": 4, "krea2.attention_gqa_f16_wmma.tokens": tokens, "krea2.attention_gqa_f16_wmma.token_capacity": capacity, "krea2.attention_gqa_f16_wmma.scale": D ** -0.5, "krea2.attention_gqa_f16_wmma.out_stride": HIDDEN}),
    ]
    for src, root, stem, cfg in jobs:
        compile_one(src, root, out / f"{stem}.hsaco", cfg)
    return out


if __name__ == "__main__":
    print(build(int(sys.argv[1])))
