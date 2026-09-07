"""The whole transformer against ComfyUI's own sampling run (tools/comfy_step.py --dump-steps).

Every evaluation is repeated on ComfyUI's latent state, timestep and text conditioning, and the
velocity is compared with the one implied by ComfyUI's denoised prediction; then our velocities
drive the same Euler steps from the same noise, and the final latent is compared.

    .venv/bin/python tools/compare_comfy_steps.py [--dumps build/comfy_parity] [--model CKPT]
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from krea2_loom import DEFAULT_MODEL


def pack(latent):
    """ComfyUI's [1][16][1][H/8][W/8] latent as the C API's [tokens][64]."""
    _, c, _, h, w = latent.shape
    x = latent.reshape(1, c, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(x, dtype=np.float32).reshape(h // 2 * (w // 2), c * 4)


def unpack(packed, c, h, w):
    x = packed.reshape(1, h // 2, w // 2, c, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    return np.ascontiguousarray(x).reshape(1, c, 1, h, w)


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", type=Path, default=ROOT / "build/comfy_parity")
    ap.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    ap.add_argument("--library", type=Path, default=ROOT / "build/libkrea2_pipeline.so")
    a = ap.parse_args()
    meta = json.loads((a.dumps / "meta.json").read_text())
    sigmas, size = meta["sigmas"], meta["size"]
    steps = len(sigmas) - 1
    text = np.load(a.dumps / "text_cond.npy")[0]                    # [tokens][12 * 2560]
    text = np.ascontiguousarray(text.reshape(text.shape[0] * 12, 2560), dtype=np.float32)
    states = [np.load(a.dumps / f"x_{i:02d}.npy") for i in range(steps)]
    denoised = [np.load(a.dumps / f"d_{i:02d}.npy") for i in range(steps)]
    channels, height, width = states[0].shape[1], states[0].shape[3], states[0].shape[4]

    lib = C.CDLL(str(a.library.resolve()))
    ptr, size_t, char = C.c_void_p, C.c_size_t, C.c_char_p
    lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, size_t]
    lib.krea2_transformer.argtypes = [ptr, ptr, size_t, C.c_int, ptr, size_t, C.c_int, C.c_int,
                                      C.c_float, ptr, size_t, char, size_t]
    lib.krea2_pipeline_destroy.argtypes = [ptr]
    lib.krea2_pipeline_destroy.restype = None
    session, error = ptr(), C.create_string_buffer(4096)

    def call(fn, *params):
        if fn(*params, error, len(error)):
            raise RuntimeError(error.value.decode())

    call(lib.krea2_pipeline_create, str(a.model.resolve()).encode(), None, C.byref(session))
    try:
        velocity = np.empty_like(pack(states[0]))
        chained = pack(np.load(a.dumps / "noise.npy"))
        print(f"{size}x{size}, {steps} steps, {text.shape[0] // 12} text tokens, model {a.model.name}")
        for i in range(steps):
            sigma = np.float32(sigmas[i])
            state = pack(states[i])
            truth = (state - pack(denoised[i])) / sigma
            call(lib.krea2_transformer, session, text.ctypes.data, text.size, text.shape[0] // 12,
                 state.ctypes.data, state.size, size, size, float(sigma),
                 velocity.ctypes.data, velocity.size)
            step = state + velocity * np.float32(sigmas[i + 1] - sigma)
            after = pack(states[i + 1]) if i + 1 < steps else pack(np.load(a.dumps / "latent_out.npy"))
            same = (f"step {i} on ComfyUI's state: velocity cosine {cosine(velocity, truth):.6f}  "
                    f"rel rms {np.linalg.norm(velocity - truth) / np.linalg.norm(truth):.4f}  "
                    f"next state cosine {cosine(step, after):.6f}")
            call(lib.krea2_transformer, session, text.ctypes.data, text.size, text.shape[0] // 12,
                 chained.ctypes.data, chained.size, size, size, float(sigma),
                 velocity.ctypes.data, velocity.size)
            drift = cosine(chained, state)
            chained = chained + velocity * np.float32(sigmas[i + 1] - sigmas[i])
            print(f"{same}  chained state cosine {drift:.6f}")
        truth = pack(np.load(a.dumps / "latent_out.npy"))
        print(f"final latent: cosine {cosine(chained, truth):.6f}  "
              f"rel rms {np.linalg.norm(chained - truth) / np.linalg.norm(truth):.4f}")
    finally:
        lib.krea2_pipeline_destroy(session)


if __name__ == "__main__":
    main()
