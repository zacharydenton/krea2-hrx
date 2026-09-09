"""Dump ComfyUI's own Krea 2 evaluation for the parity test: the packed sequence entering
block 0 (text rows then image rows, after the embedding and text fusion), the modulation
vector, the RoPE tables, and the chosen blocks' outputs at the first evaluation of a sampling
run on ComfyUI's standard loaders; optionally the sampler's per-step states.

This is the half of the parity check that cannot live in `scripts/parity.py`: it runs inside
the ComfyUI environment, against ComfyUI's own modules and loaders, where nothing else here
runs. `scripts/parity.py` compares against what it writes.

Run in the ComfyUI environment (see docs/comfyui-performance.md):
    /opt/venv/bin/python scripts/comfy_dump.py --dump-steps --out build/comfy_parity
"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy', type=Path, default=Path.home() / 'code/ComfyUI')
parser.add_argument('--models', type=Path, default=Path.home() / 'comfy-models')
parser.add_argument('--model', default='krea2_turbo_int8_convrot.safetensors')
parser.add_argument('--text-encoder', default='qwen3vl_4b_bf16.safetensors')
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--steps', type=int, default=8)
parser.add_argument('--prompt', default='a red fox in the snow')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dump-blocks', default='0,1,2,5,10,20,27')
parser.add_argument('--dump-steps', action='store_true', help='save the sampler state entering every evaluation and its denoised prediction')
parser.add_argument('--out', type=Path, default=Path('build/comfy_parity'))
args = parser.parse_args()
sys.path.insert(0, str(args.comfy))
sys.argv = [sys.argv[0], '--disable-mmap']
import comfy.options
comfy.options.enable_args_parsing()
logging.basicConfig(level=logging.WARNING)
from comfy.cli_args import args as comfy_args, enables_dynamic_vram
import comfy_aimdo.control
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
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True

out = args.out
out.mkdir(parents=True, exist_ok=True)
(out / 'blocks').mkdir(exist_ok=True)


def save(name, t):
    np.save(out / f'{name}.npy', t.detach().float().cpu().numpy())
    print('saved', name, tuple(t.shape), str(t.dtype), flush=True)


with torch.inference_mode():
    model = comfy.sd.load_diffusion_model(str(args.models / 'diffusion_models' / args.model))
    clip = comfy.sd.load_clip([str(args.models / 'text_encoders' / args.text_encoder)], clip_type=comfy.sd.CLIPType.KREA2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
    latent = torch.zeros((1, 16, 1, args.size // 8, args.size // 8))
    noise = comfy.sample.prepare_noise(latent, args.seed)
    dm = model.model.diffusion_model
    seen = set()
    meta = dict(model=args.model, text_encoder=args.text_encoder, size=args.size, steps=args.steps, prompt=args.prompt, seed=args.seed,
                model_dtype=str(model.model.get_dtype()), torch=torch.__version__)

    def hook(mod, name, pick=lambda a, kw, o: o):
        orig = mod.forward
        def f(*a, **kw):
            o = orig(*a, **kw)
            if name not in seen:
                seen.add(name)
                save(name, pick(a, kw, o))
            return o
        mod.forward = f
    blocks = [int(v) for v in args.dump_blocks.split(',') if v]
    for i in blocks:
        hook(dm.blocks[i], f'blocks/blk_{i:02d}')
    hook(dm.blocks[0], 'h_in', pick=lambda a, kw, o: a[0])
    hook(dm.blocks[0], 'tvec', pick=lambda a, kw, o: a[1])
    hook(dm.blocks[0], 'freqs', pick=lambda a, kw, o: a[2])
    hook(dm.last, 'final_in', pick=lambda a, kw, o: a[0])
    hook(dm.last, 'final_out')
    # the sigma schedule ComfyUI samples on, and the state entering each evaluation
    sigmas = comfy.samplers.simple_scheduler(model.get_model_object('model_sampling'), args.steps)
    meta['sigmas'] = [float(s) for s in sigmas]
    def cb(step, x0, x, total):
        if args.dump_steps:
            np.save(out / f'x_{step:02d}.npy', x.float().cpu().numpy())
            np.save(out / f'd_{step:02d}.npy', x0.float().cpu().numpy())
    samples = comfy.sample.sample(model, noise, args.steps, 1., 'euler', 'simple', positive, positive, latent,
                                  seed=args.seed, callback=cb, disable_pbar=True)
    np.save(out / 'latent_out.npy', samples.float().cpu().numpy())
    np.save(out / 'noise.npy', noise.float().cpu().numpy())
    cond = positive[0][0]
    meta['text_tokens'] = int(cond.shape[1])
    np.save(out / 'text_cond.npy', cond.float().cpu().numpy())
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    print('done', json.dumps({k: v for k, v in meta.items() if k != 'sigmas'}), flush=True)
