"""Train and calibrate the object-vs-seabed judge (synth/judge.py) for one (domain, seed, real %) subset.

Positives: labelled objects of the real train subset. Negatives: random seafloor boxes from the same
images, with sizes drawn from the positives and no overlap with any labelled object (3 per positive).
Model: ImageNet-pretrained ResNet18, fine-tuned. Calibration on the unseen val split: AUC, and the
threshold giving TARGET_RECALL on real val objects; the false-positive rate on val seabed at that
threshold is recorded too.

Output: /scratch/.../judge/<domain>_s{seed}_p{pct}/{judge.pt, meta.json}

Example: python scripts/train_judge.py --domain aquascan --seed 0 --real 0.25
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from synth import judge as jd  # noqa: E402
from synth.bank import read_boxes  # noqa: E402
from synth.geometry import analyze  # noqa: E402

SCRATCH = Path("/scratch/lt200304-dipmt/paweekorn")
DATA = SCRATCH / "data"
TARGET_RECALL = 0.90
NEG_PER_POS = 3


def overlaps(box, boxes, pad=4):
    x0, y0, x1, y1 = box
    return any(not (x1 + pad <= b[1] or b[3] + pad <= x0 or y1 + pad <= b[2] or b[4] + pad <= y0) for b in boxes)


def collect(domain, files, rng):
    """(crops, labels) with positives from labels and size-matched random seafloor negatives."""
    crops, labels, sizes, per_image = [], [], [], []
    for f in files:
        img = DATA / domain / "images" / f
        gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        ui = cv2.imread(str(DATA / domain / "uimask" / img.name), cv2.IMREAD_GRAYSCALE) if domain == "aquascan" else None
        swath = analyze(gray, None if ui is None else ui > 0)
        boxes = read_boxes(DATA / domain / "labels" / f"{img.stem}.txt", w, h)
        for _, x0, y0, x1, y1 in boxes:
            b = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
            if b[2] - b[0] >= 3 and b[3] - b[1] >= 3:
                crops.append(jd.crop(gray, b, swath.side((b[0] + b[2]) / 2) > 0))
                labels.append(1)
                sizes.append((b[2] - b[0], b[3] - b[1]))
        per_image.append((gray, swath, boxes))
    n_neg = NEG_PER_POS * max(1, sum(labels))
    while n_neg > 0:
        gray, swath, boxes = per_image[rng.randrange(len(per_image))]
        h, w = gray.shape
        bw, bh = sizes[rng.randrange(len(sizes))]
        ys, xs = np.nonzero(swath.seafloor[: max(1, h - bh), : max(1, w - bw)])
        if not len(xs):
            continue
        i = rng.randrange(len(xs))
        b = (int(xs[i]), int(ys[i]), int(xs[i]) + bw, int(ys[i]) + bh)
        if swath.seafloor[b[1]:b[3], b[0]:b[2]].all() and not overlaps(b, boxes):
            crops.append(jd.crop(gray, b, swath.side((b[0] + b[2]) / 2) > 0))
            labels.append(0)
            n_neg -= 1
    return crops, np.array(labels)


def auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--real", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0, help="accepted for scripts/slurm/*.sh; unused")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    pct = round(args.real * 100)
    split_dir = REPO / "data" / "splits" / args.domain
    train_files = (split_dir / ("train.txt" if pct == 100 else f"train_s{args.seed}_p{pct}.txt")).read_text().split()
    xtr, ytr = collect(args.domain, train_files, rng)
    xva, yva = collect(args.domain, (split_dir / "val.txt").read_text().split(), random.Random(1234))
    print(f"train {int(ytr.sum())} pos / {int((ytr == 0).sum())} neg, val {int(yva.sum())} pos / {int((yva == 0).sum())} neg",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = jd.build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    Xtr, Ytr = jd.to_tensor(xtr), torch.tensor(ytr, dtype=torch.float32)
    Xva = jd.to_tensor(xva)
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, ytr.sum())], device=dev)

    def scores(X):
        model.eval()
        with torch.no_grad():
            return torch.cat([torch.sigmoid(model(X[i:i + 256].to(dev)))[:, 0].cpu() for i in range(0, len(X), 256)]).numpy()

    best = (-1, None)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            x = Xtr[idx]
            if rng.random() < 0.5:
                x = x.flip(2)  # along-track flip keeps the physics
            x = x * (1 + 0.2 * (torch.rand(len(x), 1, 1, 1) - 0.5))  # brightness jitter
            loss = F.binary_cross_entropy_with_logits(model(x.to(dev))[:, 0], Ytr[idx].to(dev), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        a = auc(scores(Xva), yva)
        print(f"epoch {ep + 1} loss {loss.item():.3f} val AUC {a:.3f}", flush=True)
        if a > best[0]:
            best = (a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    s = scores(Xva)
    thr = float(np.quantile(s[yva == 1], 1 - TARGET_RECALL))
    out = SCRATCH / "judge" / f"{args.domain}_s{args.seed}_p{pct}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], out / "judge.pt")
    meta = {**vars(args), "val_auc": best[0], "threshold": thr, "target_recall": TARGET_RECALL,
            "val_recall": float((s[yva == 1] >= thr).mean()), "val_fpr": float((s[yva == 0] >= thr).mean()),
            "n_train_pos": int(ytr.sum()), "n_val_pos": int(yva.sum())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
