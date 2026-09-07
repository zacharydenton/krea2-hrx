"""The W8A8 transformer against the bf16 reference on identical inputs: latent and image PSNR.

Both arms sample the same noise with the same text states, schedule and VAE, so the only
difference is the 28 blocks: Torch bf16 on one side, the Loom W8A8 kernels through the C ABI
on the other. Three stages, one process each, because HRX and Torch cannot both take the GPU
while another job is on the box:

    .venv/bin/python tools/quality_vs_bf16.py reference   # torch: noise, text states, bf16 latents
    .venv/bin/python tools/quality_vs_bf16.py native      # HRX only: the same steps in W8A8
    .venv/bin/python tools/quality_vs_bf16.py compare     # torch: tiled VAE decode and PSNR

--work holds the shared files (default build/quality). Run the stages in that order; the
reference stage only has to be repeated when the prompt, seed, size or step count change.
"""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))


def scheduler_sigma(step: int, steps: int, mu: float = 1.15) -> np.float32:
    """host/native_schedule.h, which the scheduler regression pins against diffusers."""
    if step == steps:
        return np.float32(0)
    raw = np.float32(1.0 - step / steps)
    shift = np.float32(math.exp(mu))
    return shift / (shift + (np.float32(1) / raw - np.float32(1)))


def job(work: Path) -> dict:
    return json.loads((work / "job.json").read_text())


def reference(a) -> None:
    """Torch, the bf16 checkpoint: write the shared inputs and this arm's latents."""
    import torch
    from tools.pipeline import build, TURBO

    pipe = build("none", None, checkpoint=Path(a.checkpoint or TURBO), distilled=True)
    with torch.no_grad():
        embeds, mask = pipe.encode_prompt(a.prompt, device="cuda")
        noise = pipe.prepare_latents(1, 16, a.size, a.size, torch.bfloat16, "cuda",
                                     torch.Generator("cuda").manual_seed(a.seed))
        a.work.mkdir(parents=True, exist_ok=True)
        # The native transformer takes the valid text rows only; the reference masks the rest.
        text = embeds[0][mask[0].bool()].float().cpu().numpy()
        np.save(a.work / "text.npy", np.ascontiguousarray(text))
        np.save(a.work / "noise.npy", np.ascontiguousarray(noise[0].float().cpu().numpy()))
        out = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask, latents=noise,
                   height=a.size, width=a.size, num_inference_steps=a.steps,
                   guidance_scale=0.0, output_type="latent")
    torch.save(out.images.cpu(), a.work / "bf16.pt")
    (a.work / "job.json").write_text(json.dumps(
        dict(prompt=a.prompt, seed=a.seed, size=a.size, steps=a.steps,
             checkpoint=str(a.checkpoint or TURBO), text_tokens=int(text.shape[0])), indent=1))
    print(f"reference: {text.shape[0]} text tokens, latents {tuple(out.images.shape)} -> {a.work}/bf16.pt")


def native(a) -> None:
    """HRX only (Torch on the CPU for bf16 rounding): the same steps through the W8A8 blocks."""
    import torch
    from krea2_loom import DEFAULT_MODEL

    meta = job(a.work)
    text = np.ascontiguousarray(np.load(a.work / "text.npy").reshape(-1, 2560), np.float32)
    latents = np.ascontiguousarray(np.load(a.work / "noise.npy"), np.float32)
    size, steps = meta["size"], meta["steps"]
    lib = C.CDLL(str(ROOT / "build/libkrea2_pipeline.so"))
    ptr, usize, char = C.c_void_p, C.c_size_t, C.c_char_p
    lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, usize]
    lib.krea2_transformer.argtypes = [ptr, ptr, usize, C.c_int, ptr, usize, C.c_int, C.c_int,
                                      C.c_float, ptr, usize, char, usize]
    lib.krea2_pipeline_destroy.argtypes = [ptr]
    lib.krea2_pipeline_destroy.restype = None
    session, error = ptr(), C.create_string_buffer(4096)

    def call(fn, *params):
        if fn(*params, error, len(error)):
            raise RuntimeError(error.value.decode())

    call(lib.krea2_pipeline_create, str(Path(a.weights or DEFAULT_MODEL).resolve()).encode(),
         None, C.byref(session))
    try:
        velocity = np.empty_like(latents)
        state = torch.from_numpy(latents).bfloat16()          # the pipeline's own bf16 state
        for step in range(steps):
            sigma, following = scheduler_sigma(step, steps), scheduler_sigma(step + 1, steps)
            current = np.ascontiguousarray(state.float().numpy())
            call(lib.krea2_transformer, session, text.ctypes.data, text.size, text.shape[0] // 12,
                 current.ctypes.data, current.size, size, size, float(sigma),
                 velocity.ctypes.data, velocity.size)
            # kernels/native/euler.loom: bf16 delta, bf16 product, bf16 sum.
            delta = torch.tensor(following - sigma).bfloat16().float()
            product = (delta * torch.from_numpy(velocity).bfloat16().float()).bfloat16().float()
            state = (state.float() + product).bfloat16()
            print(f"  step {step}: sigma {sigma:.6f}", flush=True)
    finally:
        lib.krea2_pipeline_destroy(session)
    torch.save(state.float()[None], a.work / "w8a8.pt")
    print(f"native: latents -> {a.work}/w8a8.pt")


def compare(a) -> None:
    import subprocess
    meta = job(a.work)
    subprocess.run([sys.executable, str(ROOT / "tools/decode_latents.py"),
                    str(a.work / "bf16.pt"), str(a.work / "w8a8.pt"), "--size", str(meta["size"])],
                   check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["reference", "native", "compare"])
    ap.add_argument("--work", type=Path, default=ROOT / "build/quality")
    ap.add_argument("--prompt", default="a red fox sitting in fresh snow at dawn, soft light, photograph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--checkpoint", default=None, help="bf16 checkpoint for the reference arm")
    ap.add_argument("--weights", default=None, help="int8 ConvRot checkpoint for the native arm")
    a = ap.parse_args()
    {"reference": reference, "native": native, "compare": compare}[a.stage](a)


if __name__ == "__main__":
    main()
