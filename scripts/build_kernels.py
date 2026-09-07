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


# The GEMM launch shape rules, mirrored from host/gemm_shape.h; the launch metadata
# records their results and the session rejects a bundle that disagrees with its own.
WIDE_TILE_TOKENS = 2048


def gemm_pitch(k, bits=4):
    """Operand row pitch in k elements: an 8192-byte row (K = 16384) aliases in the cache, so it
    gets one 64-byte k step of padding (down projection 61 -> 77 TOPS); 3072-byte rows showed no effect."""
    return k + 512 // bits if (k * bits // 8) % 8192 == 0 else k


def gemm_grid_rows(tokens, rows, m_group):
    """Launch grid rows: the tiles themselves for the shortening 256-row kernels, padded to whole
    raster groups for the 128-row kernels."""
    tiles = (tokens + rows - 1) // rows
    return tiles if rows == 256 else (tiles + m_group - 1) // m_group * m_group


def gemm_rows(tokens, bits=4):
    """Workgroup tile rows: the 256x128 tile runs 4-11% faster per row (paired A/B, 2064..16896
    tokens) but rounds M up to 256, so it is chosen from WIDE_TILE_TOKENS when its rows are within
    8% of the 128-row grid's padded rows (at 1040 tokens it lost 3%). The int8 family only has it."""
    if bits == 8:
        return 256
    if tokens < WIDE_TILE_TOKENS:
        return 128
    wide = (tokens + 255) // 256
    narrow = gemm_grid_rows(tokens, 128, gemm_m_group(tokens, 128))
    return 256 if 50 * wide <= 27 * narrow else 128


def gemm_m_group(tokens, rows=128):
    """m-tiles per raster group. The 256-row kernels shorten their last group in-kernel and always
    take 4; the 128-row kernels pad the grid: 1 for a single tile row, else of 4, 3, 2 the one that
    pads the tile rows least (ties to the larger). Persisted beside the kernels for the host."""
    if rows == 256:
        return 4
    tiles = (tokens + rows - 1) // rows
    if tiles == 1:
        return 1
    return min((4, 3, 2), key=lambda g: ((tiles + g - 1) // g * g, -g))


def fp16_query_tiles(tokens):
    """query32 fails trajectory quality; retain the original fp16 kernel."""
    return 1


def build(tokens: int, bits: int = 4) -> Path:
    """Return a complete, immutable kernel bundle matching the current sources and configuration.
    bits: the GEMM operand width the weights were exported for (build/weights/config.json)."""
    if not 16 <= tokens <= 16896:
        raise ValueError("tokens must be 16..16896")
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    waves = 8 if tokens < 8192 else 4
    attention_bits = int(os.environ.get("KREA2_ATTN_QK") or 16)   # 16: f16 QK and PV (ComfyUI's SDPA class); 4 / 8: the smoothed int4 / int8 QK kernels
    if attention_bits not in (4, 8, 16):
        raise ValueError("KREA2_ATTN_QK must be 4, 8 or 16")
    query_tiles = fp16_query_tiles(tokens) if attention_bits == 16 else 1
    source = ("attention_query32" if query_tiles == 2 else "attention_gqa_lds_f16_wmma") if attention_bits == 16 else f"attention_sage_i{attention_bits}_fast" + ("" if waves == 8 else "_prefetch")
    rows = gemm_rows(tokens, bits)
    m_group = gemm_m_group(tokens, rows)
    pitch_hidden, pitch_inter = gemm_pitch(HIDDEN, bits), gemm_pitch(INTER, bits)
    ib = f"i{bits}"
    capacity = max((tokens + 16 + 31) // 32 * 32, (tokens + 63) // 64 * 64)   # tokens+16 headroom, whole 64-key blocks, a multiple of 32
    if query_tiles == 2:
        capacity = (tokens + 16 + 63) // 64 * 64
    def gemm(kind, stem, k, n):
        source = f"gemm_{ib}{kind}" + ("_256" if rows == 256 else "")
        ns = f"krea2.{source}"
        return (source, f"krea2_{source}", stem, {f"{ns}.k_size": k, f"{ns}.k_stride": gemm_pitch(k, bits), f"{ns}.n_size": n, f"{ns}.m_group": m_group})
    def prepare(kind, width, pitch, **extra):
        source = f"prepare_{kind}_{ib}"
        ns = f"krea2.{source}"
        return (source, f"krea2_{source}", source, {f"{ns}.width": width, f"{ns}.out_stride": pitch, **{f"{ns}.{key}": value for key, value in extra.items()}})
    jobs = [
        prepare("norm", HIDDEN, pitch_hidden, eps=1e-5),
        prepare("gated", HIDDEN, pitch_hidden, gate_stride=QKVG),
        prepare("plain", INTER, pitch_inter),
        gemm("", "gemm_qkvg", HIDDEN, QKVG),
        gemm("_swiglu", "gemm_gu", HIDDEN, 2 * INTER),
        gemm("_resid", "gemm_wo", HIDDEN, HIDDEN),
        gemm("_resid", "gemm_down", INTER, HIDDEN),
        ("rope_qknorm_f16", "krea2_rope_qknorm_f16", "rope_qknorm", {"krea2.rope_qknorm_f16.row_stride": QKVG, "krea2.rope_qknorm_f16.q_heads": 48, "krea2.rope_qknorm_f16.kv_heads": KV, "krea2.rope_qknorm_f16.k_offset": HIDDEN, "krea2.rope_qknorm_f16.eps": 1e-5}),
        (source, "krea2_" + source, "attention",
         {f"krea2.{source}.{key}": value for key, value in
          dict(q_stride=HIDDEN, kv_stride=KV * D, tokens=tokens,
               token_capacity=capacity, scale=D ** -0.5, out_stride=HIDDEN).items()}),
    ]
    if query_tiles == 2:
        jobs.append(("sage_transpose", "krea2_sage_transpose", "attention_transpose",
                     {"krea2.sage_transpose.width": KV * D,
                      "krea2.sage_transpose.row_capacity": capacity}))
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
    launch = f"5 {tokens} {rows} {m_group} {capacity} {waves} {pitch_hidden} {pitch_inter} {attention_bits} {bits} {query_tiles}\n"
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
    print(build(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4))
