"""Time repeated C-API generation calls; no Torch or model framework is imported.

The first generation includes block-session preparation. Subsequent generations
reuse the same session. Run without KREA2_NATIVE_PROFILE for timing comparisons.
If a repeat changes, both lossless images and a report are saved under build/.
"""

import argparse
import os
import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent


def save_mismatch(first, current, args, run):
    """Retain the evidence before the reused output buffer is destroyed."""
    root = ROOT / "build"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="native-mismatch-", dir=root))
    header = f"P6\n{args.size} {args.size}\n255\n".encode()
    (output / "first.ppm").write_bytes(header + first)
    (output / "repeat.ppm").write_bytes(header + current)
    difference = [abs(a - b) for a, b in zip(first, current)]
    report = {
        "library": str(args.library.resolve()),
        "bundle": str(args.bundle.resolve()),
        "prompt": args.prompt,
        "width": args.size,
        "height": args.size,
        "steps": args.steps,
        "seed": 0,
        "first_run": 0,
        "repeat_run": run,
        "first_rgb_sha256": hashlib.sha256(first).hexdigest(),
        "repeat_rgb_sha256": hashlib.sha256(current).hexdigest(),
        "changed_channels": sum(d != 0 for d in difference),
        "changed_pixels": sum(
            any(difference[i : i + 3]) for i in range(0, len(difference), 3)
        ),
        "max_absolute_error": max(difference),
        "mean_absolute_error": sum(difference) / len(difference),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, default=ROOT / "build/libkrea2_pipeline.so")
    ap.add_argument("--bundle", "--model", dest="bundle", type=Path,
                    default=Path(os.environ.get("KREA2_MODEL") or Path.home() / "comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors"),
                    help="ComfyUI's int8 ConvRot checkpoint (text encoder and VAE found beside it)")
    ap.add_argument("--prompt", default="a red fox in the snow")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    if args.runs < 1 or not 64 <= args.size <= 2048 or args.size % 16:
        ap.error("runs must be positive and size a multiple of 16 in 64..2048")
    lib = C.CDLL(str(args.library.resolve()))
    ptr, size, char = C.c_void_p, C.c_size_t, C.c_char_p
    lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, size]
    lib.krea2_pipeline_destroy.argtypes = [ptr]
    lib.krea2_pipeline_destroy.restype = None
    lib.krea2_generate.argtypes = [
        ptr,
        char,
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_uint64,
        ptr,
        size,
        ptr,
        size,
        char,
        size,
    ]
    session, error = ptr(), C.create_string_buffer(4096)
    rgb = (C.c_uint8 * (args.size * args.size * 3))()

    def call(fn, *params):
        if fn(*params, error, len(error)):
            raise RuntimeError(error.value.decode())

    start = time.perf_counter()
    call(
        lib.krea2_pipeline_create,
        str(args.bundle.resolve()).encode(),
        None,
        C.byref(session),
    )
    print(json.dumps({"load_seconds": time.perf_counter() - start}), flush=True)
    try:
        first_digest = None
        first_rgb = None
        for run in range(args.runs):
            start = time.perf_counter()
            call(
                lib.krea2_generate,
                session,
                args.prompt.encode(),
                args.size,
                args.size,
                args.steps,
                0,
                None,
                0,
                rgb,
                len(rgb),
            )
            elapsed = time.perf_counter() - start
            current = bytes(rgb)
            digest = hashlib.sha256(current).hexdigest()
            print(
                json.dumps({"run": run, "seconds": elapsed, "rgb_sha256": digest}),
                flush=True,
            )
            if first_digest is not None and digest != first_digest:
                output = save_mismatch(first_rgb, current, args, run)
                raise RuntimeError(
                    "identical prompt and seed produced different RGB on a repeated call; "
                    f"images and error report: {output}"
                )
            if first_digest is None:
                first_digest, first_rgb = digest, current
    finally:
        lib.krea2_pipeline_destroy(session)


if __name__ == "__main__":
    main()
