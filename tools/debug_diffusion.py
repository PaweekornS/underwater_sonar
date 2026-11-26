"""Sanity check for the diffusion pilot on CPU: is the flat grey fill caused by the LoRA training or by the
pipeline? Rows: [masked | base SD-inpainting (no LoRA) | with LoRA | real], on unseen val windows.

Usage: python tools/debug_diffusion.py <domain> <seed> <real> <out.jpg>
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image  # diffusers reads numpy inputs as [0, 1] floats; pass PIL

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HOME", "/scratch/lt200304-dipmt/paweekorn/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline  # noqa: E402

from synth import inpaint as ip  # noqa: E402
from synth.bank import read_boxes  # noqa: E402
from synth.geometry import analyze  # noqa: E402

domain, seed, real, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
NAMES = {"cylinder2": ["cylinder", "manta"], "mine": ["MILCO", "NOMBO"], "aquascan": ["human"]}
S = Path("/scratch/lt200304-dipmt/paweekorn")
D = S / "data" / domain

windows = []
for f in (REPO / "data" / "splits" / domain / "val.txt").read_text().split():
    g = cv2.imread(str(D / "images" / f), cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    sw = analyze(g)
    for c, x0, y0, x1, y1 in read_boxes(D / "labels" / f"{Path(f).stem}.txt", w, h):
        b = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
        sb = sw.side((b[0] + b[2]) / 2) > 0
        win = ip.window(b, w, h)
        windows.append((ip.crop(g, win, sb), ip.mask_in_window(b, win, sb), c))
    if len(windows) >= 4:
        break

pipe = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting",
                                                      variant="fp16", torch_dtype=torch.float32, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)


def run(img, mask, c):
    r = pipe(prompt=ip.prompt(domain, NAMES[domain][c]), image=Image.fromarray(img).convert("RGB"),
             mask_image=Image.fromarray((mask * 255).astype(np.uint8)), height=ip.RES, width=ip.RES, num_inference_steps=20,
             guidance_scale=4.0, generator=torch.Generator().manual_seed(0), output_type="np").images[0].mean(2)
    return (mask * r * 255 + (1 - mask) * img).clip(0, 255).astype(np.uint8)


base = [run(*wd) for wd in windows[:4]]
print("base done", flush=True)
pipe.load_lora_weights(S / "inpaint_lora" / f"{domain}_s{seed}_p{round(real * 100)}",
                       weight_name="pytorch_lora_weights.safetensors")
lora = [run(*wd) for wd in windows[:4]]
rows = [np.hstack([(img * (1 - m) + 128 * m).astype(np.uint8), b, l, img])
        for (img, m, _), b, l in zip(windows, base, lora)]
cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
print("saved", out, flush=True)
