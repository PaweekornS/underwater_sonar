"""LoRA fine-tuning of Stable Diffusion 1.5 inpainting on one domain's real train subset (box-conditioned).

Each object in the subset gives one example: its context window, a mask over the box plus its far-range
shadow side, and a class prompt. The model learns to redraw a masked object consistent with the
surrounding seabed. Only objects from that seed's subset are used (no val/test leakage).

Output: /scratch/.../inpaint_lora/<domain>_s{seed}_p{pct}/ (diffusers LoRA weights + meta.json)

Example: python scripts/train_inpaint.py --domain aquascan --seed 0 --real 0.25
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from synth import inpaint as ip  # noqa: E402
from synth.bank import read_boxes  # noqa: E402
from synth.geometry import analyze  # noqa: E402

SCRATCH = Path("/scratch/lt200304-dipmt/paweekorn")
DATA = SCRATCH / "data"
os.environ.setdefault("HF_HOME", str(SCRATCH / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
BASE = "stable-diffusion-v1-5/stable-diffusion-inpainting"
NAMES = {"cylinder2": ["cylinder", "manta"], "mine": ["MILCO", "NOMBO"], "aquascan": ["human"]}


def ui_mask(domain, image):
    if domain != "aquascan":
        return None
    m = cv2.imread(str(DATA / domain / "uimask" / image.name), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def build_examples(domain, files):
    """(image path, box, class, starboard) for every labelled object in the subset."""
    out = []
    for f in files:
        img = DATA / domain / "images" / f
        gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        swath = analyze(gray, ui_mask(domain, img))
        for c, x0, y0, x1, y1 in read_boxes(DATA / domain / "labels" / f"{img.stem}.txt", w, h):
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                out.append((img, (x0, y0, x1, y1), c, swath.side((x0 + x1) / 2) > 0))
    return out


def sample(ex, rng):
    """One augmented (image, mask) pair in [-1, 1] / {0, 1}, canonical orientation."""
    img, box, c, starboard = ex
    gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    wx, wy, side = ip.window(box, w, h)
    jitter = int(0.15 * side)  # shift the object inside the window so position is not fixed
    wx = int(np.clip(wx + rng.randint(-jitter, jitter), 0, w - side))
    wy = int(np.clip(wy + rng.randint(-jitter, jitter), 0, h - side))
    win = (wx, wy, side)
    im = ip.crop(gray, win, starboard).astype(np.float32) / 127.5 - 1
    m = ip.mask_in_window(box, win, starboard)
    if rng.random() < 0.5:  # along-track flip keeps the physics (shadow still points right)
        im, m = im[::-1].copy(), m[::-1].copy()
    return im, m, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(NAMES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--real", type=float, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0, help="accepted for scripts/slurm/array.sh; unused")
    args = ap.parse_args()

    from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    pct = round(args.real * 100)
    split = REPO / "data" / "splits" / args.domain / ("train.txt" if pct == 100 else f"train_s{args.seed}_p{pct}.txt")
    examples = build_examples(args.domain, split.read_text().split())
    out = SCRATCH / "inpaint_lora" / f"{args.domain}_s{args.seed}_p{pct}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(examples)} objects -> {out}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(BASE, variant="fp16", torch_dtype=torch.float16,
                                                          safety_checker=None)
    vae, text_encoder, tokenizer = pipe.vae.to(dev), pipe.text_encoder.to(dev), pipe.tokenizer
    unet = pipe.unet.to(dev, dtype=torch.float32)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = DDPMScheduler.from_config(pipe.scheduler.config)

    with torch.no_grad():
        texts = [ip.prompt(args.domain, n) for n in NAMES[args.domain]] + [""]
        tok = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        emb = text_encoder(tok.input_ids.to(dev))[0].float()  # rows: classes..., then empty (CFG dropout)

    for step in range(1, args.steps + 1):
        batch = [sample(examples[rng.randrange(len(examples))], rng) for _ in range(args.batch)]
        im = torch.tensor(np.stack([b[0] for b in batch]))[:, None].repeat(1, 3, 1, 1).to(dev)
        m = torch.tensor(np.stack([b[1] for b in batch]))[:, None].to(dev)
        cls = torch.tensor([b[2] if rng.random() > 0.1 else len(NAMES[args.domain]) for b in batch], device=dev)
        with torch.no_grad():
            lat = vae.encode(im.half()).latent_dist.sample().float() * vae.config.scaling_factor
            masked = vae.encode((im * (1 - m)).half()).latent_dist.sample().float() * vae.config.scaling_factor
        m64 = F.interpolate(m, size=lat.shape[-2:], mode="nearest")
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev)
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(torch.cat([noisy, m64, masked], 1), t, encoder_hidden_states=emb[cls]).sample
        # weight the masked region more: that is where the object must be drawn
        w = 1.0 + 4.0 * m64
        loss = (w * (pred.float() - noise) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

    StableDiffusionInpaintPipeline.save_lora_weights(out, unet_lora_layers=get_peft_model_state_dict(unet))
    (out / "meta.json").write_text(json.dumps({**vars(args), "n_objects": len(examples), "base": BASE}, indent=2))


if __name__ == "__main__":
    main()
