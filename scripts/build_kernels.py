"""Compile the block kernels into build/kernels/T<tokens>/<fingerprint>/.

    python3 scripts/build_kernels.py <tokens>

The bundle includes immutable launch metadata. Its fingerprint covers the sources,
builder, compiler identity and configuration; the printed path is ready for krea2_create."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIDDEN, KV, D, INTER = 6144, 12, 128, 16384
QKVG = HIDDEN + 2 * KV * D + HIDDEN


def compiler() -> Path:
    return Path(os.environ.get("LOOM_COMPILE") or Path.home() / "code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile").resolve()


def compile_one(src: str, root: str, out: Path, cfg: dict) -> None:
    loom = str(compiler())
    cmd = [loom, str(ROOT / "kernels" / f"{src}.loom"), "--backend=amdgpu-hal", "--target=gfx1151", f"--root=@{root}", f"--output={out}"]
    cmd += [f"--config={k}={v}" for k, v in cfg.items()]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{src}: {r.stderr.strip()[:600]}")


def gemm_m_group(tokens):
    """m-tiles per raster group: of 4, 3, 2 the one that pads the tile rows least (ties to the larger).
    Persisted beside the kernels for the host to use throughout the session."""
    tiles = (tokens + 127) // 128
    return min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))


def build(tokens: int) -> Path:
    """Return a complete, immutable kernel bundle matching the current sources and configuration."""
    if not 16 <= tokens <= 16896:
        raise ValueError("tokens must be 16..16896")
    waves = 8 if tokens < 8192 else 4
    source = "attention_sage_i4_fast" if waves == 8 else "attention_sage_i4_fast_prefetch"
    m_group = gemm_m_group(tokens)
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 63) // 64 * 64)   # tokens+16 headroom, whole 64-key blocks, a multiple of 32
    jobs = [
        ("prepare_norm_i4", "krea2_prepare_norm_i4", "prepare_norm_i4", {"krea2.prepare_norm_i4.width": HIDDEN, "krea2.prepare_norm_i4.eps": 1e-5}),
        ("prepare_gated_i4", "krea2_prepare_gated_i4", "prepare_gated_i4", {"krea2.prepare_gated_i4.width": HIDDEN, "krea2.prepare_gated_i4.gate_stride": QKVG}),
        ("prepare_plain_i4", "krea2_prepare_plain_i4", "prepare_plain_i4", {"krea2.prepare_plain_i4.width": INTER}),
        ("gemm_i4", "krea2_gemm_i4", "gemm_qkvg", {"krea2.gemm_i4.k_size": HIDDEN, "krea2.gemm_i4.n_size": QKVG, "krea2.gemm_i4.m_group": m_group}),
        ("gemm_i4_swiglu", "krea2_gemm_i4_swiglu", "gemm_gu", {"krea2.gemm_i4_swiglu.k_size": HIDDEN, "krea2.gemm_i4_swiglu.n_size": 2 * INTER, "krea2.gemm_i4_swiglu.m_group": m_group}),
        ("gemm_i4_resid", "krea2_gemm_i4_resid", "gemm_wo", {"krea2.gemm_i4_resid.k_size": HIDDEN, "krea2.gemm_i4_resid.n_size": HIDDEN, "krea2.gemm_i4_resid.m_group": m_group}),
        ("gemm_i4_resid", "krea2_gemm_i4_resid", "gemm_down", {"krea2.gemm_i4_resid.k_size": INTER, "krea2.gemm_i4_resid.n_size": HIDDEN, "krea2.gemm_i4_resid.m_group": m_group}),
        ("rope_qknorm_f16", "krea2_rope_qknorm_f16", "rope_qknorm", {"krea2.rope_qknorm_f16.row_stride": QKVG, "krea2.rope_qknorm_f16.q_heads": 48, "krea2.rope_qknorm_f16.kv_heads": KV, "krea2.rope_qknorm_f16.k_offset": HIDDEN, "krea2.rope_qknorm_f16.eps": 1e-5}),
        (source, "krea2_" + source, "attention",
         {f"krea2.{source}.{key}": value for key, value in
          dict(q_stride=HIDDEN, kv_stride=KV * D, tokens=tokens,
               token_capacity=capacity, scale=D ** -0.5, out_stride=HIDDEN).items()}),
    ]
    loom = compiler()
    stat = loom.stat()
    signature = dict(jobs=jobs, compiler=[str(loom), stat.st_size, stat.st_mtime_ns],
                     builder=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     sources={src: hashlib.sha256((ROOT / "kernels" / f"{src}.loom").read_bytes()).hexdigest()
                              for src, _, _, _ in jobs})
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    parent = ROOT / "build/kernels" / f"T{tokens}"
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / fingerprint
    launch = f"2 {tokens} {m_group} {capacity} {waves}\n"
    # Serialize publication, including across Python processes. A failed compilation
    # never exposes a partial bundle or overwrites kernels used by a live session.
    with (parent / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.exists():
            if (out / "launch.txt").read_text() != launch:
                raise RuntimeError(f"invalid kernel launch metadata: {out}")
            hashes = json.loads((out / "manifest.json").read_text())
            for _, _, stem, _ in jobs:
                name = f"{stem}.hsaco"
                if hashlib.sha256((out / name).read_bytes()).hexdigest() != hashes.get(name):
                    raise RuntimeError(f"corrupt cached kernel: {out / name}")
            return out
        with tempfile.TemporaryDirectory(prefix=".compile-", dir=parent) as tmp:
            staging = Path(tmp) / "bundle"
            staging.mkdir()
            for src, root, stem, cfg in jobs:
                compile_one(src, root, staging / f"{stem}.hsaco", cfg)
            (staging / "launch.txt").write_text(launch)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in staging.glob("*.hsaco")}
            (staging / "manifest.json").write_text(json.dumps(hashes, sort_keys=True))
            staging.rename(out)
    return out


if __name__ == "__main__":
    print(build(int(sys.argv[1])))
