"""GPU scheduler, modulation, and resolution reuse checks without a Torch model.

Requires scripts/build_native.sh and build/native-deploy. Torch is only an oracle.
"""
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(os.environ.get("KREA2_TEST_BIN", ROOT / "build"))
sys.path.insert(0, str(ROOT))
from tools.pipeline import SCHEDULER


def scheduler_check(root):
    rng = np.random.default_rng(29)
    samples = rng.normal(size=(5050, 256)).astype(np.float32)
    velocity = (rng.normal(size=samples.shape) * 32).astype(np.float32)
    samples.tofile(root / "samples.bin")
    velocity.tofile(root / "velocity.bin")
    subprocess.run([str(BIN / "krea2-scheduler-test"), str(root)], check=True)
    got = np.fromfile(root / "result.bin", np.float32).reshape(samples.shape)
    x = torch.from_numpy(samples).cuda().bfloat16()
    v = torch.from_numpy(velocity).cuda().bfloat16()
    expected = torch.empty_like(x)
    native_times = torch.from_numpy(np.fromfile(root / "sigmas.bin", np.float32)).cuda().bfloat16()
    row = 0
    regression_detected = False
    for steps in range(1, 101):
        scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER)
        scheduler.set_timesteps(sigmas=np.linspace(1., 1. / steps, steps), mu=1.15, device="cuda")
        assert scheduler.sigmas.is_cuda
        assert torch.equal(native_times[row:row + steps], (scheduler.timesteps / 1000).bfloat16())
        for step, t in enumerate(scheduler.timesteps):
            expected[row] = scheduler.step(v[row], t, x[row], return_dict=False)[0]
            # The old fp32-delta path must actually differ on this fixture.
            delta = scheduler.sigmas[step + 1] - scheduler.sigmas[step]
            old = (x[row].float() + (delta * v[row].float()).bfloat16().float()).bfloat16()
            regression_detected |= not torch.equal(expected[row], old)
            row += 1
    assert regression_detected, "fixture failed to expose the scheduler rounding bug"
    np.testing.assert_array_equal(got, expected.float().cpu().numpy())
    print("PASS CUDA scheduler: all 5,050 steps across step counts 1..100, exact bf16 outputs", flush=True)


def modulation_check(root, bundle):
    rng = np.random.default_rng(33)
    rng.normal(size=(12, 2560)).astype(np.float32).tofile(root / "text.bin")
    rng.normal(size=(16, 6144)).astype(np.float32).tofile(root / "hidden.bin")
    subprocess.run([str(BIN / "krea2-native-components"), str(bundle), str(root)], check=True)
    mod = torch.from_numpy(np.fromfile(root / "mod.bin", np.float32)).cuda().bfloat16()
    meta = json.loads((bundle / "transformer/weights.json").read_text())
    expected = []
    with (bundle / "transformer/weights.bin").open("rb") as weights:
        for i in range(28):
            entry = meta[f"blocks.{i}.mod.lin"]
            weights.seek(entry["offset"])
            raw = np.frombuffer(weights.read(entry["bytes"]), np.uint16).copy()
            table = torch.from_numpy(raw).view(torch.bfloat16).cuda()
            expected.append((mod + table).float().cpu().numpy())
    got = np.fromfile(root / "block_mod.bin", np.float32).reshape(28, -1)
    np.testing.assert_array_equal(got, np.stack(expected))
    print("PASS device modulation: all 28 tables exactly match bf16 Torch addition", flush=True)


def pipeline_check(root, bundle):
    lib = C.CDLL(str(BIN / "libkrea2_pipeline.so"))
    ptr, size, char = C.c_void_p, C.c_size_t, C.c_char_p
    lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, size]
    lib.krea2_pipeline_destroy.argtypes = [ptr]
    lib.krea2_pipeline_destroy.restype = None
    lib.krea2_encode.argtypes = [ptr, char, ptr, size, C.POINTER(size), char, size]
    lib.krea2_transformer.argtypes = [ptr, ptr, size, C.c_int, ptr, size, C.c_int, C.c_int, C.c_float, ptr, size, char, size]
    lib.krea2_decode.argtypes = [ptr, ptr, size, C.c_int, C.c_int, ptr, size, char, size]
    lib.krea2_generate.argtypes = [ptr, char, C.c_int, C.c_int, C.c_int, C.c_uint64, ptr, size, ptr, size, char, size]
    session, error = ptr(), C.create_string_buffer(4096)
    # Missing JSON files must identify the exact path through the C ABI.
    missing = root / "missing"
    assert lib.krea2_pipeline_create(os.fsencode(missing), None, C.byref(session), error, len(error)) != 0
    assert os.fsencode(missing / "native.json") in error.value and not session
    missing.mkdir()
    (missing / "native.json").write_text('{"version":1,"model":"krea2-turbo"}')
    assert lib.krea2_pipeline_create(os.fsencode(missing), None, C.byref(session), error, len(error)) != 0
    assert os.fsencode(missing / "text/weights.json") in error.value and not session
    print("PASS missing bundle JSON errors include file paths", flush=True)

    # Only private symlinks are removed; source bundle files are untouched.
    private = root / "bundle"
    private.mkdir()
    for name in ("native.json", "text", "transformer", "vae", "tokenizer.json", "sources", "kernels"):
        (private / name).symlink_to(bundle / name)
    (private / "blocks").mkdir()
    for name in ("weights.bin", "manifest.txt"):
        (private / "blocks" / name).symlink_to(bundle / "blocks" / name)
    def call(fn, *args):
        if fn(*args, error, len(error)):
            raise RuntimeError(error.value.decode())
    call(lib.krea2_pipeline_create, os.fsencode(private), None, C.byref(session))
    try:
        prompt = b"a red fox in the snow"
        count = size()
        call(lib.krea2_encode, session, prompt, None, 0, C.byref(count))
        taps = np.empty((count.value, 12, 2560), np.float32)
        call(lib.krea2_encode, session, prompt, taps.ctypes.data, taps.size, C.byref(count))
        rng = np.random.default_rng(17)
        initial = rng.normal(size=(16, 64)).astype(np.float32)
        rectangle = rng.normal(size=(32, 64)).astype(np.float32)
        def forward(latents, w, h, timestep):
            out = np.empty_like(latents)
            call(lib.krea2_transformer, session, taps.ctypes.data, taps.size, count.value,
                 latents.ctypes.data, latents.size, w, h, timestep, out.ctypes.data, out.size)
            return out
        first = forward(initial, 64, 64, .75)
        (private / "blocks/weights.bin").unlink()
        (private / "blocks/manifest.txt").unlink()
        forward(rectangle, 64, 128, .75)
        np.testing.assert_array_equal(forward(initial, 64, 64, .75), first)
        print("PASS 64x64 -> 64x128 -> 64x64: exact output, no weight file available after first call", flush=True)
        scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER)
        scheduler.set_timesteps(sigmas=np.linspace(1., .5, 2), mu=1.15, device="cuda")
        current = torch.from_numpy(initial).cuda().bfloat16()
        for t in scheduler.timesteps:
            velocity = forward(current.float().cpu().numpy(), 64, 64, float((t / 1000).bfloat16()))
            current = scheduler.step(torch.from_numpy(velocity).cuda().bfloat16(), t, current, return_dict=False)[0]
        final = current.float().cpu().numpy()
        expected = np.empty((64, 64, 3), np.uint8)
        call(lib.krea2_decode, session, final.ctypes.data, final.size, 64, 64, expected.ctypes.data, expected.size)
        got = np.empty_like(expected)
        call(lib.krea2_generate, session, prompt, 64, 64, 2, 0, initial.ctypes.data, initial.size, got.ctypes.data, got.size)
        np.testing.assert_array_equal(got, expected)
        print("PASS two-step native generation vs components driven by CUDA diffusers scheduler: exact RGB", flush=True)
    finally:
        lib.krea2_pipeline_destroy(session)


def main():
    bundle = (ROOT / "build/native-deploy").resolve()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with torch.no_grad():
            scheduler_check(root)
            modulation_check(root, bundle)
            pipeline_check(root, bundle)


if __name__ == "__main__":
    main()
