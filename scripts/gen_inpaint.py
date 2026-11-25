"""Generate synthetic images by box-conditioned inpainting, with verification.

For each synthetic image: take a real background from the subset, choose boxes (class, size and count
from the subset's distribution; positions on seafloor, not overlapping), inpaint an object into each box
with the domain's LoRA, then verify every new object:
  - signal: the box must stand out from the seabed (signal-to-noise >= MIN_SNR), and
  - detector: a YOLO11n trained on the same real subset must find it (IoU >= 0.5 with any prediction).
Failed objects are restored to the original pixels and regenerated, up to MAX_TRIES; still-failing ones
are dropped. The acceptance rate is written to stats.json.

Output: DATA/<domain>/synth/inpaint[-noverify]/s{seed}_p{pct}/{images,labels,stats.json}

Example: python scripts/gen_inpaint.py --domain aquascan --seed 0 --real 0.25 --count-ratio 1
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from synth import inpaint as ip  # noqa: E402
from synth.bank import read_boxes  # noqa: E402
from synth.geometry import analyze  # noqa: E402

SCRATCH = Path("/scratch/lt200304-dipmt/paweekorn")
DATA = SCRATCH / "data"
os.environ.setdefault("HF_HOME", str(SCRATCH / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
NAMES = {"cylinder2": ["cylinder", "manta"], "mine": ["MILCO", "NOMBO"], "aquascan": ["human"]}
MIN_SNR = 1.0
MAX_TRIES = 3


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(NAMES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--real", type=float, required=True)
    ap.add_argument("--count-ratio", type=float, default=1.0)
    ap.add_argument("--no-verify", action="store_true", help="ablation: keep every generated object")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=0, help="accepted for scripts/slurm/array.sh; unused")
    args = ap.parse_args()

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
    from ultralytics import YOLO

    pct = round(args.real * 100)
    split = REPO / "data" / "splits" / args.domain / ("train.txt" if pct == 100 else f"train_s{args.seed}_p{pct}.txt")
    real = [DATA / args.domain / "images" / f for f in split.read_text().split()]
    name = "inpaint" + ("-noverify" if args.no_verify else "")
    out = DATA / args.domain / "synth" / name / f"s{args.seed}_p{pct}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # Subset statistics: objects per positive image, class frequencies, box sizes per (class, image size).
    counts, classes, sizes = [], [], {}
    for im in real:
        h, w = cv2.imread(str(im), cv2.IMREAD_GRAYSCALE).shape
        boxes = read_boxes(DATA / args.domain / "labels" / f"{im.stem}.txt", w, h)
        if boxes:
            counts.append(len(boxes))
        for c, x0, y0, x1, y1 in boxes:
            classes.append(c)
            sizes.setdefault((c, f"{w}x{h}"), []).append((x1 - x0, y1 - y0))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", variant="fp16", torch_dtype=torch.float16,
        safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(SCRATCH / "inpaint_lora" / f"{args.domain}_s{args.seed}_p{pct}")
    pipe.set_progress_bar_config(disable=True)
    detector = None
    if not args.no_verify:
        run = f"{args.domain}_r{pct}_none_augon_yolon_s{args.seed}"
        detector = YOLO(str(SCRATCH / "runs" / run / "train" / "weights" / "best.pt"))
    gen = torch.Generator("cuda").manual_seed(args.seed)

    def inpaint(gray, box, c, starboard):
        h, w = gray.shape
        win = ip.window(box, w, h)
        img = ip.crop(gray, win, starboard)
        mask = ip.mask_in_window(box, win, starboard)
        res = pipe(prompt=ip.prompt(args.domain, NAMES[args.domain][c]), image=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB),
                   mask_image=(mask * 255).astype(np.uint8), height=ip.RES, width=ip.RES,
                   num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen,
                   output_type="np").images[0]
        ip.uncrop(gray, (res.mean(2) * 255), mask, win, starboard)
        return win

    n = math.ceil(len(real) * args.count_ratio)
    tried = accepted = 0
    for i in range(n):
        rng = random.Random(args.seed * 1_000_003 + i)
        bg = real[rng.randrange(len(real))]
        gray = cv2.imread(str(bg), cv2.IMREAD_GRAYSCALE)
        original = gray.copy()
        h, w = gray.shape
        ui = cv2.imread(str(DATA / args.domain / "uimask" / bg.name), cv2.IMREAD_GRAYSCALE) if args.domain == "aquascan" else None
        swath = analyze(gray, None if ui is None else ui > 0)
        existing = read_boxes(DATA / args.domain / "labels" / f"{bg.stem}.txt", w, h)
        occupied = np.zeros((h, w), bool)
        for _, x0, y0, x1, y1 in existing:
            occupied[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = True

        # choose boxes: class from subset frequency, size from same class and image-size group, on seafloor
        pending = []
        for _ in range(counts[rng.randrange(len(counts))] if counts else 1):
            c = classes[rng.randrange(len(classes))]
            pool = sizes.get((c, f"{w}x{h}")) or [s for (cc, _), v in sizes.items() if cc == c for s in v]
            bw, bh = pool[rng.randrange(len(pool))]
            ys, xs = np.nonzero(swath.seafloor[: h - bh, : w - bw])
            for _ in range(100):
                if not len(xs):
                    break
                j = rng.randrange(len(xs))
                x, y = int(xs[j]), int(ys[j])
                # keep the whole inpainting window clear of other objects, so they are never redrawn
                wx, wy, side = ip.window((x, y, x + bw, y + bh), w, h)
                if swath.seafloor[y:y + bh, x:x + bw].all() and not occupied[wy:wy + side, wx:wx + side].any():
                    occupied[wy:wy + side, wx:wx + side] = True
                    pending.append(((x, y, x + bw, y + bh), c, swath.side(x + bw / 2) > 0))
                    break

        new = []
        for attempt in range(1 if args.no_verify else MAX_TRIES):
            if not pending:
                break
            wins = [inpaint(gray, box, c, sb) for box, c, sb in pending]
            tried += len(pending)
            if args.no_verify:
                new += [(c, *box) for box, c, _ in pending]
                accepted += len(pending)
                break
            preds = detector.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), imgsz=1024, conf=0.05,
                                     verbose=False)[0].boxes.xyxy.cpu().numpy()
            still = []
            for (box, c, sb), (wx, wy, side) in zip(pending, wins):
                ok = ip.signal_to_noise(gray, box) >= MIN_SNR and any(iou(box, p) >= 0.5 for p in preds)
                if ok:
                    new.append((c, *box))
                    accepted += 1
                else:  # undo this window and try again
                    gray[wy:wy + side, wx:wx + side] = original[wy:wy + side, wx:wx + side]
                    still.append((box, c, sb))
            pending = still

        cv2.imwrite(str(out / "images" / f"syn_{i:05d}.png"), gray)
        lines = [f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"
                 for c, x0, y0, x1, y1 in existing + new]
        (out / "labels" / f"syn_{i:05d}.txt").write_text("".join(lines))
        if i % 50 == 0:
            print(f"{i}/{n} attempts {tried} accepted {accepted}", flush=True)

    stats = {"images": n, "inpaint_attempts": tried, "accepted": accepted,
             "acceptance_rate": accepted / max(tried, 1), "verify": not args.no_verify, **vars(args)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
