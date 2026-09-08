"""Time resident native VAE decodes on fixed packed latents; verify repeated RGB.

Run zero warms the session and may compile kernels; exclude it from warm timing.
Model loading is outside the timer. Use --output to compare RGB between builds.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("latents", type=Path, help="packed float32 .npy latents")
    ap.add_argument("--library", type=Path, default=ROOT / "build/libkrea2.so")
    ap.add_argument("--model", type=Path, default=Path.home() / "comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--output", type=Path, help="save the first RGB buffer for exact A/B comparison")
    ap.add_argument("--profile", action="store_true", help="profile kernels after the first decode; timings are then diagnostic")
    a = ap.parse_args()
    os.environ.pop("KREA2_NATIVE_KERNEL_PROFILE", None)
    if a.runs < 1:
        ap.error("runs must be positive")
    latents = np.ascontiguousarray(np.load(a.latents), np.float32)
    rgb = np.empty((a.height, a.width, 3), np.uint8)
    lib = C.CDLL(str(a.library.resolve()))
    ptr, size, char = C.c_void_p, C.c_size_t, C.c_char_p
    lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, size]
    lib.krea2_pipeline_destroy.argtypes = [ptr]
    lib.krea2_pipeline_destroy.restype = None
    lib.krea2_decode.argtypes = [ptr, ptr, size, C.c_int, C.c_int, ptr, size, char, size]
    session, error = ptr(), C.create_string_buffer(4096)
    def call(fn, *args):
        if fn(*args, error, len(error)):
            raise RuntimeError(error.value.decode())
    call(lib.krea2_pipeline_create, os.fsencode(a.model.resolve()), None, C.byref(session))
    print(json.dumps(dict(library=str(a.library.resolve()), width=a.width, height=a.height,
                          binary_sha256={name: hashlib.sha256((a.library.resolve().parent / name).read_bytes()).hexdigest()
                                         for name in (a.library.name, "libkrea2.so")},
                          latents_sha256=hashlib.sha256(latents.tobytes()).hexdigest(),
                          profile=a.profile)), flush=True)
    first = None
    try:
        for run in range(a.runs):
            if a.profile and run == 1:
                os.environ["KREA2_NATIVE_KERNEL_PROFILE"] = "1"
            start = time.perf_counter()
            call(lib.krea2_decode, session, latents.ctypes.data, latents.size,
                 a.width, a.height, rgb.ctypes.data, rgb.size)
            seconds = time.perf_counter() - start
            raw = rgb.tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            print(json.dumps(dict(run=run, seconds=seconds, rgb_sha256=digest)), flush=True)
            if first is None:
                first = raw
                if a.output:
                    a.output.write_bytes(raw)
            elif first != raw:
                raise RuntimeError("repeated VAE decode changed RGB")
    finally:
        lib.krea2_pipeline_destroy(session)


if __name__ == "__main__":
    main()
