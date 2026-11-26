"""Generate synthetic training images for one (domain, seed, real fraction, method) cell.

The object bank, backgrounds, object counts, and object sizes all come from that seed's real train
subset only, so low-data runs never see objects outside their own subset.

Output: DATA/<domain>/synth/<name>/s{seed}_p{pct}/{images,labels}, where name is the method, plus
"-no<component>" for a leave-one-out ablation. scripts/train.py reads the same path.

Example:
  python scripts/make_synth.py --domain mine --seed 0 --real 0.25 --method ours --count-ratio 4
"""
import argparse
import math
import random
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from synth.bank import extract, read_boxes  # noqa: E402
from synth.compose import COMPONENTS, Options, compose  # noqa: E402

DATA = Path("/scratch/lt200304-dipmt/paweekorn/data")
HAS_UI = {"aquascan"}


def ui_mask(domain, image):
    if domain not in HAS_UI:
        return None
    m = cv2.imread(str(DATA / domain / "uimask" / image.name), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def label_of(image):
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def bank_one(domain, image):
    return extract(image, label_of(image), ui_mask(domain, image))


def make_one(i, domain, method, opt, seed, backgrounds, bank, counts, sizes, out):
    rng = random.Random(seed * 1_000_003 + i)
    bg = backgrounds[rng.randrange(len(backgrounds))]
    gray = cv2.imread(str(bg), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    res = f"{w}x{h}"
    # Mine has two sensors/resolutions: only paste objects from the same image-size group.
    pool = [o for o in bank if o.res == res] or bank
    size_pool = sizes.get(res) or [s for v in sizes.values() for s in v]
    objs = [pool[rng.randrange(len(pool))] for _ in range(counts[rng.randrange(len(counts))])]
    existing = read_boxes(label_of(bg), w, h)
    img, labels = compose(gray, existing, objs, method, rng, opt, size_pool, ui_mask(domain, bg))
    name = f"syn_{i:05d}"
    cv2.imwrite(str(out / "images" / f"{name}.png"), img)
    lines = []
    for c, x0, y0, x1, y1 in labels:
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            lines.append(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
    (out / "labels" / f"{name}.txt").write_text("".join(lines))
    return len(labels) - len(existing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--real", type=float, required=True, choices=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--method", required=True,
                    help='naive | ours | box-<steps> with steps from synth.compose.BOX_STEPS, e.g. box-sea-mirror')
    ap.add_argument("--ablate", default="", choices=["", *COMPONENTS, "featherblend"],
                    help="leave one component out; 'featherblend' = Poisson blending instead of feathering")
    ap.add_argument("--count-ratio", type=float, default=1.0, help="synthetic images per real train image")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pct = round(args.real * 100)
    split = REPO / "data" / "splits" / args.domain / ("train.txt" if pct == 100 else f"train_s{args.seed}_p{pct}.txt")
    real = [DATA / args.domain / "images" / f for f in split.read_text().split()]
    name = args.method + (f"-no{args.ablate}" if args.ablate else "")
    out = DATA / args.domain / "synth" / name / f"s{args.seed}_p{pct}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        bank = [o for objs in pool.map(partial(bank_one, args.domain), real) for o in objs]

    counts, sizes = [], {}
    for im in real:
        w, h = (lambda a: (a.shape[1], a.shape[0]))(cv2.imread(str(im), cv2.IMREAD_GRAYSCALE))
        boxes = read_boxes(label_of(im), w, h)
        if boxes:
            counts.append(len(boxes))
        sizes.setdefault(f"{w}x{h}", []).extend((x1 - x0, y1 - y0) for _, x0, y0, x1, y1 in boxes)

    opt = Options(off={args.ablate} if args.ablate in COMPONENTS else set(),
                  blend="poisson" if args.ablate == "featherblend" else "feather")
    n = math.ceil(len(real) * args.count_ratio)
    job = partial(make_one, domain=args.domain, method=args.method, opt=opt, seed=args.seed, backgrounds=real,
                  bank=bank, counts=counts or [1], sizes=sizes, out=out)
    with Pool(args.workers) as pool:
        pasted = pool.map(job, range(n))
    print(f"{out}: {n} images, {sum(pasted)} objects pasted, bank {len(bank)} "
          f"({sum(o.usable for o in bank)} usable), real {len(real)}")


if __name__ == "__main__":
    main()
