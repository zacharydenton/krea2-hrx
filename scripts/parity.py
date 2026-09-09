#!/usr/bin/env python3
"""Parity against ComfyUI: does this host compute what ComfyUI's Krea 2 does?

Self-contained — the C ABI binding, the latent packing and the checks are all in this file, and
numpy is the only import that is not in the standard library. It is deliberately not part of
`scripts/test.sh`: it needs the checkpoint, a GPU, and a directory of dumps produced inside the
ComfyUI environment by `scripts/comfy_dump.py`. Run it before a release, not on every change.

    python3 scripts/parity.py gate [--require]   the release gate: this host against ComfyUI's run
    python3 scripts/parity.py steps              every evaluation, verbose
    python3 scripts/parity.py image              the whole pipeline on ComfyUI's noise, to PNG

Everything else in this repository checks that the model agrees with its own earlier output.
This is the only check against something outside itself, which is why it is worth keeping even
though it cannot run unattended.
"""
import argparse
import ctypes as C
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path.home() / "comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors"

# What the gate demands of a full run from ComfyUI's noise. The last recorded measurement
# (9235e13, the 1024x1024 eight-step Turbo dump) puts the final latent at cosine 0.806, so this
# gate does not pass today and is not expected to: it is the standing statement of what parity
# with ComfyUI means, not a description of where the host currently is.
#
# The per-evaluation velocity is much closer -- cosine 0.9981 on the first evaluation and
# 0.9989-0.9999 after it, relative rms falling from 6.2% to 1.4% -- and is reported beside the
# gate because it localizes a failure. A velocity that stays close while the latent diverges is
# the Euler step cancelling: at sigma 0.311 it subtracts two large terms, so a 1.4% velocity
# difference becomes a 55% latent difference. That explains the gap; it does not excuse it, and
# the thresholds stay on the number that says whether we produce ComfyUI's image.
MIN_COSINE = 0.99
MAX_RELATIVE_RMS = 0.15

# ComfyUI returns the sampler's latent with the Wan latent format applied (x * std + mean); the
# pipeline's own decoder applies it, so undo it before comparing against the sampler's state.
LATENT_MEAN = np.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
                       np.float32).reshape(1, 16, 1, 1, 1)
LATENT_STD = np.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
                      np.float32).reshape(1, 16, 1, 1, 1)


def sampler_latent(processed):
    return (processed - LATENT_MEAN) / LATENT_STD


def pack(latent):
    """ComfyUI's [1][16][1][H/8][W/8] latent as the C API's [tokens][64]."""
    _, c, _, h, w = latent.shape
    x = latent.reshape(1, c, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(x, dtype=np.float32).reshape(h // 2 * (w // 2), c * 4)


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def relative_rms(ours, truth):
    return float(np.linalg.norm(ours - truth) / np.linalg.norm(truth))


def png_rgb(path, width, height, rgb):
    """A PNG without Pillow, so the parity check needs nothing beyond numpy."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class Pipeline:
    """The pipeline C ABI through ctypes: create, transformer, decode, generate, destroy."""

    def __init__(self, library, model):
        self.lib = C.CDLL(str(Path(library).resolve()))
        ptr, size, char, i32, f32 = C.c_void_p, C.c_size_t, C.c_char_p, C.c_int, C.c_float
        self.lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, size]
        self.lib.krea2_pipeline_destroy.argtypes = [ptr]
        self.lib.krea2_pipeline_destroy.restype = None
        self.lib.krea2_transformer.argtypes = [ptr, ptr, size, i32, ptr, size, i32, i32, f32,
                                               ptr, size, char, size]
        self.lib.krea2_decode.argtypes = [ptr, ptr, size, i32, i32, ptr, size, char, size]
        self.lib.krea2_generate.argtypes = [ptr, char, i32, i32, i32, C.c_uint64, ptr, size,
                                            ptr, size, char, size]
        self.error = C.create_string_buffer(4096)
        self.handle = ptr()
        self.call(self.lib.krea2_pipeline_create, str(Path(model).resolve()).encode(), None,
                  C.byref(self.handle))

    def call(self, fn, *params):
        if fn(*params, self.error, len(self.error)):
            raise RuntimeError(self.error.value.decode())

    def transformer(self, text, text_tokens, state, width, height, sigma, velocity):
        self.call(self.lib.krea2_transformer, self.handle, text.ctypes.data, text.size,
                  text_tokens, state.ctypes.data, state.size, width, height, float(sigma),
                  velocity.ctypes.data, velocity.size)
        return velocity

    def decode(self, latents, width, height):
        rgb = (C.c_uint8 * (width * height * 3))()
        self.call(self.lib.krea2_decode, self.handle, latents.ctypes.data, latents.size,
                  width, height, rgb, len(rgb))
        return bytes(rgb)

    def generate(self, prompt, width, height, steps, seed, noise):
        rgb = (C.c_uint8 * (width * height * 3))()
        self.call(self.lib.krea2_generate, self.handle, prompt.encode(), width, height, steps,
                  seed, noise.ctypes.data, noise.size, rgb, len(rgb))
        return bytes(rgb)

    def close(self):
        if self.handle:
            self.lib.krea2_pipeline_destroy(self.handle)
            self.handle = C.c_void_p()


class Dump:
    """One `scripts/comfy_dump.py --dump-steps` directory."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.sigmas = self.meta["sigmas"]
        self.steps = len(self.sigmas) - 1
        self.size = self.meta["size"]
        text = np.load(self.dir / "text_cond.npy")[0]              # [tokens][12 * 2560]
        self.text_tokens = text.shape[0]
        self.text = np.ascontiguousarray(text.reshape(text.shape[0] * 12, 2560), dtype=np.float32)
        missing = [f"x_{i:02d}.npy" for i in range(self.steps) if not (self.dir / f"x_{i:02d}.npy").is_file()]
        if missing:
            raise SystemExit(f"{self.dir} has no sampler states ({missing[0]} and others): "
                             "re-run scripts/comfy_dump.py with --dump-steps")
        self.states = [np.load(self.dir / f"x_{i:02d}.npy") for i in range(self.steps)]
        self.denoised = [np.load(self.dir / f"d_{i:02d}.npy") for i in range(self.steps)]
        self.noise = np.load(self.dir / "noise.npy")
        self.final = sampler_latent(np.load(self.dir / "latent_out.npy"))


def chained(pipeline, dump, verbose):
    """Every evaluation twice: on ComfyUI's own state, and on our own chained state.

    The first is the measurement that carries signal -- our transformer against ComfyUI's, on
    identical inputs. The second accumulates, and tells a drift where it started: a low velocity
    cosine is the transformer, a high one beside a diverging chained state is the sampler.

    Returns the chained final latent and the per-evaluation velocity agreement.
    """
    state = pack(dump.noise)
    velocity = np.empty_like(state)
    agreement = []
    for i in range(dump.steps):
        sigma, step = dump.sigmas[i], np.float32(dump.sigmas[i + 1] - dump.sigmas[i])
        theirs = pack(dump.states[i])
        truth = (theirs - pack(dump.denoised[i])) / np.float32(sigma)
        pipeline.transformer(dump.text, dump.text_tokens, theirs, dump.size, dump.size,
                             sigma, velocity)
        similarity, error = cosine(velocity, truth), relative_rms(velocity, truth)
        agreement.append((similarity, error))
        if verbose:
            after = pack(dump.states[i + 1]) if i + 1 < dump.steps else dump.final_packed
            print(f"  step {i} sigma {sigma:.4f}: velocity cosine {similarity:.6f}"
                  f"  rel rms {error:.4f}"
                  f"  next state cosine {cosine(theirs + velocity * step, after):.6f}"
                  f"  chained state cosine {cosine(state, theirs):.6f}", flush=True)
        pipeline.transformer(dump.text, dump.text_tokens, state, dump.size, dump.size,
                             sigma, velocity)
        state = state + velocity * step
    return state, agreement


def open_pipeline(a):
    library = Path(a.library)
    if not library.is_file():
        raise SystemExit(f"{library} does not exist: run scripts/build.sh first")
    return Pipeline(library, a.model)


def command_gate(a):
    """`gate` and `steps` are the same run; `steps` also prints every evaluation."""
    dump = Dump(a.dumps)
    dump.final_packed = pack(dump.final)
    print(f"{dump.size}x{dump.size}, {dump.steps} steps, {dump.text_tokens} text tokens, "
          f"seed {dump.meta['seed']}, ComfyUI {dump.meta['model']} in {dump.meta['model_dtype']}")
    pipeline = open_pipeline(a)
    try:
        ours, agreement = chained(pipeline, dump, verbose=a.command == "steps")
    finally:
        pipeline.close()
    # Reported beside the gate because it localizes a failure, never in place of it.
    worst_cosine = min(similarity for similarity, _ in agreement)
    worst_rms = max(error for _, error in agreement)
    print(f"velocity on ComfyUI's states: worst cosine {worst_cosine:.6f}  "
          f"worst rel rms {worst_rms:.4f}")
    similarity, error = cosine(ours, dump.final_packed), relative_rms(ours, dump.final_packed)
    ok = similarity >= a.min_cosine and error <= a.max_rms
    print(f"final latent: cosine {similarity:.6f} (>= {a.min_cosine})  "
          f"rel rms {error:.4f} (<= {a.max_rms})  {'PASS' if ok else 'FAIL'}")
    if not ok and a.require:
        return 1
    return 0


def command_image(a):
    """The whole pipeline on ComfyUI's noise, beside ComfyUI's own latent through our decoder.

    Two PNGs that should look like the same image: differences visible here but not in `gate`
    are the decoder's, and differences in both are the transformer's.
    """
    dump = Dump(a.dumps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = open_pipeline(a)
    try:
        theirs = pipeline.decode(pack(dump.final), dump.size, dump.size)
        png_rgb(out / "comfy_latent.png", dump.size, dump.size, theirs)
        ours = pipeline.generate(dump.meta["prompt"], dump.size, dump.size, dump.steps,
                                 dump.meta["seed"], pack(dump.noise))
        png_rgb(out / "ours.png", dump.size, dump.size, ours)
    finally:
        pipeline.close()
    a8, b8 = np.frombuffer(ours, np.uint8).astype(np.float64), np.frombuffer(theirs, np.uint8).astype(np.float64)
    mse = float(np.mean((a8 - b8) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"wrote {out}/ours.png sha256 {hashlib.sha256(ours).hexdigest()}")
    print(f"wrote {out}/comfy_latent.png sha256 {hashlib.sha256(theirs).hexdigest()}")
    print(f"our pipeline against ComfyUI's latent through our decoder: PSNR {psnr:.2f} dB")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["gate", "steps", "image"])
    ap.add_argument("--dumps", type=Path, default=ROOT / "build/comfy_parity",
                    help="a scripts/comfy_dump.py --dump-steps directory")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--library", type=Path, default=ROOT / "build/libkrea2.so")
    ap.add_argument("--out", type=Path, default=ROOT / "build/parity")
    ap.add_argument("--min-cosine", type=float, default=MIN_COSINE)
    ap.add_argument("--max-rms", type=float, default=MAX_RELATIVE_RMS)
    ap.add_argument("--require", action="store_true",
                    help="exit non-zero when the gate's thresholds are not met")
    a = ap.parse_args()
    if not Path(a.dumps).is_dir():
        raise SystemExit(f"{a.dumps} does not exist: produce it with scripts/comfy_dump.py "
                         "inside the ComfyUI environment")
    return {"gate": command_gate, "steps": command_gate, "image": command_image}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
