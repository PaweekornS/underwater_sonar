"""Debug the generative pilots on CPU: (1) verification pass rate of *unseen* real objects (val split), as the
fair reference for the generated objects' acceptance rate; (2) a contact sheet of diffusion and GAN
inpaintings on val windows: [masked | diffusion | GAN | real].

Usage: python tools/debug_inpaint.py <domain> <seed> <real> <out.jpg>
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
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("HF_HOME", "/scratch/lt200304-dipmt/paweekorn/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import train_gan_inpaint as tg  # noqa: E402
from gen_inpaint import NAMES, iou  # noqa: E402
from synth import inpaint as ip  # noqa: E402
from synth.bank import read_boxes  # noqa: E402
from synth.geometry import analyze  # noqa: E402

domain, seed, real, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
pct = round(real * 100)
S = Path("/scratch/lt200304-dipmt/paweekorn")
D = S / "data" / domain
tag = f"{domain}_s{seed}_p{pct}"

from ultralytics import YOLO  # noqa: E402
det = YOLO(str(S / "runs" / f"{domain}_r{pct}_none_augon_yolon_s{seed}" / "train" / "weights" / "best.pt"))
val = (REPO / "data" / "splits" / domain / "val.txt").read_text().split()
n = ok = 0
objs = []
for f in val:
    g = cv2.imread(str(D / "images" / f), cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    preds = det.predict(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), imgsz=1024, conf=0.05, verbose=False,
                        device="cpu")[0].boxes.xyxy.numpy()
    sw = analyze(g)
    for c, x0, y0, x1, y1 in read_boxes(D / "labels" / f"{Path(f).stem}.txt", w, h):
        b = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
        n += 1
        ok += ip.signal_to_noise(g, b) >= 1.0 and any(iou(b, p) >= 0.5 for p in preds)
        objs.append((g, b, c, sw.side((b[0] + b[2]) / 2) > 0))
print(f"VAL (unseen) real objects: n={n} pass={ok / max(n, 1):.2f}", flush=True)

from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline  # noqa: E402
pipe = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting",
                                                      variant="fp16", torch_dtype=torch.float32, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(S / "inpaint_lora" / tag, weight_name="pytorch_lora_weights.safetensors")
pipe.set_progress_bar_config(disable=True)
n_cls = len(NAMES[domain])
G = tg.load_networks().define_G(2 + n_cls, 1, 64, "unet_256", norm="batch", use_dropout=True)
G.load_state_dict(torch.load(S / "inpaint_gan" / tag / "generator.pt", map_location="cpu"))
G.eval()

rows = []
for g, b, c, sb in objs[:6]:
    h, w = g.shape
    win = ip.window(b, w, h)
    img, mask = ip.crop(g, win, sb), ip.mask_in_window(b, win, sb)
    dif = pipe(prompt=ip.prompt(domain, NAMES[domain][c]), image=Image.fromarray(img).convert("RGB"),
               mask_image=Image.fromarray((mask * 255).astype(np.uint8)), height=ip.RES, width=ip.RES, num_inference_steps=20,
               guidance_scale=4.0, generator=torch.Generator().manual_seed(0), output_type="np").images[0].mean(2)
    r = tg.RES
    im_t = torch.tensor(cv2.resize(img, (r, r)) / 127.5 - 1, dtype=torch.float32)[None, None]
    m_t = torch.tensor(cv2.resize(mask, (r, r), interpolation=cv2.INTER_NEAREST))[None, None]
    with torch.no_grad():
        gan = cv2.resize(((G(tg.gen_input(im_t, m_t, torch.tensor([c]), n_cls))[0, 0].numpy() + 1) / 2), (ip.RES, ip.RES))
    comp = lambda x: (mask * x * 255 + (1 - mask) * img).clip(0, 255).astype(np.uint8)
    masked = (img * (1 - mask) + 128 * mask).astype(np.uint8)
    rows.append(np.hstack([masked, comp(dif), comp(gan), img]))
    print("row done", flush=True)
cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
