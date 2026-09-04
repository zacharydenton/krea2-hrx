"""Decode saved packed latents with the tiled VAE (the plain 1024^2 decode runs for
many minutes on this stack) and compare runs: PSNR in latent space and in pixels.

    python3 tools/decode_latents.py build/bf16.pt build/w4a4.pt --size 1024"""
import argparse
from pathlib import Path

import numpy as np
import torch

MODELS = Path.home() / "krea2-models"


def unpack(latents, height, width, vae_scale=8, p=2):
    b, _, c = latents.shape
    h, w = p * (height // (vae_scale * p)), p * (width // (vae_scale * p))
    x = latents.view(b, h // p, w // p, c // (p * p), p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c // (p * p), 1, h, w)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("latents", nargs="+"); ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    from diffusers import AutoencoderKLQwenImage
    from PIL import Image
    vae = AutoencoderKLQwenImage.from_pretrained(str(MODELS / "qwen-image" / "vae"), torch_dtype=torch.bfloat16).to("cuda")
    vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).cuda().to(torch.bfloat16)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).cuda().to(torch.bfloat16)
    images, lats = {}, {}
    for path in a.latents:
        lat = torch.load(path).cuda()
        lats[path] = lat.float()
        z = unpack(lat.to(torch.bfloat16), a.size, a.size) / std + mean
        with torch.no_grad():
            img = vae.decode(z, return_dict=False)[0][:, :, 0]
        img = ((img.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        images[path] = img
        out = Path(path).with_suffix(".png"); Image.fromarray(img).save(out); print(f"decoded {path} -> {out}")
    if len(a.latents) == 2:
        p0, p1 = a.latents
        d = (lats[p0] - lats[p1]).pow(2).mean().item(); sig = lats[p0].pow(2).mean().item()
        print(f"latent PSNR {10 * np.log10(sig / max(d, 1e-30)):.2f} dB  (rel rms {np.sqrt(d / sig):.4f})")
        e = ((images[p0].astype(np.float64) - images[p1].astype(np.float64)) ** 2).mean()
        print(f"image PSNR {10 * np.log10(255 ** 2 / max(e, 1e-30)):.2f} dB")


if __name__ == "__main__":
    main()
