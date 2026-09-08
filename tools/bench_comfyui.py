"""Benchmark local ComfyUI's standard Krea2 loaders, Euler sampler and VAE.

Run in the ComfyUI environment. No server, extensions, previews or image encoding;
all inference stages execute on every run, bypassing graph-output caching.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy', type=Path, default=Path.home() / 'code/ComfyUI')
parser.add_argument('--models', type=Path, default=Path.home() / 'comfy-models')
parser.add_argument('--model', default='krea2_turbo_int8_convrot.safetensors')
parser.add_argument('--size', type=int, default=1024, help='square, unless --width/--height are given')
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
parser.add_argument('--steps', type=int, default=8)
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--prompt', default='a red fox in the snow')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output-dir', type=Path, help='Save PNG, packed initial noise and timing metadata')
args = parser.parse_args()
sys.path.insert(0, str(args.comfy))
# The local Strix Halo toolbox recommends disabling checkpoint mmap.
sys.argv = [sys.argv[0], '--disable-mmap']
import comfy.options
comfy.options.enable_args_parsing()
logging.basicConfig(level=logging.INFO)
from comfy.cli_args import args as comfy_args, enables_dynamic_vram
import comfy_aimdo.control
# Match main.py's ROCm startup and default DynamicVRAM initialization.
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
if enables_dynamic_vram():
    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not comfy_args.disable_nvml_pressure)
import torch
import comfy.sd
import comfy.sample
import comfy.model_management
import comfy.utils
import comfy.memory_management
import comfy.model_patcher
if enables_dynamic_vram() and comfy.model_management.rocm_version >= (7, 14):
    if comfy_aimdo.control.init_devices((d.index, int(comfy_args.vram_headroom * 1024**3))
                                       for d in comfy.model_management.get_all_torch_devices()):
        comfy_aimdo.control.set_log_info()
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True


def stamp():
    torch.cuda.synchronize()
    return time.perf_counter()


args.width = args.width or args.size
args.height = args.height or args.size

with torch.inference_mode():
    start = stamp()
    model = comfy.sd.load_diffusion_model(str(args.models / 'diffusion_models' / args.model))
    clip = comfy.sd.load_clip([str(args.models / 'text_encoders/qwen3vl_4b_fp8_scaled.safetensors')],
                              clip_type=comfy.sd.CLIPType.KREA2)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.models / 'vae/qwen_image_vae.safetensors')))
    metadata = dict(load_seconds=stamp() - start, torch=torch.__version__,
                          model=args.model, model_dtype=str(model.model.get_dtype()),
                          size=args.size, width=args.width, height=args.height,
                          steps=args.steps, cfg=1, sampler='euler',
                          scheduler='simple', disable_mmap=True,
                          dynamic_vram=comfy.memory_management.aimdo_enabled)
    print(json.dumps(metadata), flush=True)
    latent = torch.zeros((1, 16, 1, args.height // 8, args.width // 8))
    noise = comfy.sample.prepare_noise(latent, args.seed)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Same [image tokens, channels * 2 * 2] packing as the native C API.
        packed = noise.reshape(1, 16, args.height // 16, 2, args.width // 16, 2)
        packed = packed.permute(0, 2, 4, 1, 3, 5).contiguous().numpy().astype('<f4')
        packed.tofile(args.output_dir / 'noise.bin')
        metadata['noise_sha256'] = hashlib.sha256(packed.tobytes()).hexdigest()
    for run in range(args.runs):
        start = stamp()
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
        encoded = stamp()
        samples = comfy.sample.sample(model, noise, args.steps, 1., 'euler', 'simple',
                                      positive, positive, latent, seed=args.seed, disable_pbar=True)
        sampled = stamp()
        image = vae.decode(samples)
        decoded = stamp()
        rgb = (image.clamp(0, 1) * 255).byte().cpu().numpy().tobytes()
        result = dict(run=run, text_seconds=encoded - start,
                              denoise_seconds=sampled - encoded, vae_seconds=decoded - sampled,
                              seconds=decoded - start, cached_prompt_seconds=decoded - encoded,
                              rgb_sha256=hashlib.sha256(rgb).hexdigest(),
                              peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        print(json.dumps(result), flush=True)
        if args.output_dir:
            from PIL import Image
            Image.frombytes('RGB', (args.size, args.size), rgb).save(args.output_dir / 'comfy.png')
            (args.output_dir / 'comfy.json').write_text(json.dumps({**metadata, **result}, indent=2))
