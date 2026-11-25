"""Train one detector run and write results/<run_id>.json.

Every config sees the same number of optimizer steps: each epoch is a fixed-size
"virtual epoch" of EPOCH_IMAGES images, built by repeating the shuffled train list.
The checkpoint is selected on real val only; the final metrics are on real test.

Example:
  python scripts/train.py --domain mine --real 0.25 --synth none --aug on --model yolo --seed 3
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/scratch/lt200304-dipmt/paweekorn/data")
RUNS = Path("/scratch/lt200304-dipmt/paweekorn/runs")
WEIGHTS = Path("/scratch/lt200304-dipmt/paweekorn/weights")
MODELS = {"yolo": "yolo11s.pt", "yolon": "yolo11n.pt", "rtdetr": "rtdetr-l.pt"}
NAMES = {"cylinder2": ["cylinder", "manta"], "mine": ["MILCO", "NOMBO"], "mine_leaky": ["MILCO", "NOMBO"], "aquascan": ["human"]}

AUG_OFF = dict(hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0, scale=0, shear=0, perspective=0,
               flipud=0, fliplr=0, mosaic=0, mixup=0, cutmix=0, copy_paste=0, erasing=0)


def read_list(path):
    return [line for line in Path(path).read_text().splitlines() if line]


def link_split(files, dst):
    """Symlink images and their labels into a per-run folder. Ultralytics writes labels.cache next to the
    label folder, so runs sharing the data folder would race on one cache file when packed on a GPU."""
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    links = {}
    for f in files:
        if f in links:
            continue
        src = Path(f)
        name = f"{len(links):06d}_{src.name}"
        img = dst / "images" / name
        lab = dst / "labels" / f"{Path(name).stem}.txt"
        for link, target in ((img, src), (lab, src.parent.parent / "labels" / f"{src.stem}.txt")):
            if not link.is_symlink():
                link.symlink_to(target)
        links[f] = str(img)
    return [links[f] for f in files]


def train_images(domain, real, seed):
    split_dir = REPO / "data" / "splits" / domain
    name = "train.txt" if real == 1.0 else f"train_s{seed}_p{round(real * 100)}.txt"
    return [str(DATA / domain / "images" / f) for f in read_list(split_dir / name)]


def synth_images(domain, synth, real, seed, ratio, n_real):
    """Synthetic images live in DATA/<domain>/synth/<method>/s{seed}_p{pct}/images (built by synth/)."""
    if synth == "none":
        return []
    root = DATA / domain / "synth" / synth / f"s{seed}_p{round(real * 100)}" / "images"
    files = sorted(str(p) for p in root.iterdir())
    need = round(n_real * ratio)
    if len(files) < need:
        raise SystemExit(f"need {need} synthetic images in {root}, found {len(files)}")
    return random.Random(seed).sample(files, need)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(NAMES), required=True)
    ap.add_argument("--real", type=float, default=1.0, choices=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--synth", default="none", help="none | naive | ours | box-<steps> (synth folder name)")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--aug", default="on", choices=["on", "off"])
    ap.add_argument("--model", default="yolo", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="", help="synth component left out, e.g. shadow (selects synth dir)")
    ap.add_argument("--synth-only", action="store_true",
                    help="diagnostic: train on the synthetic images alone (val/test stay real)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-images", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.synth != "none" and args.aug == "off":
        raise SystemExit("synthetic configs are stacked on standard augmentation (aug=on)")
    synth_name = args.synth + (f"-no{args.ablate}" if args.ablate else "")
    run_id = (f"{args.domain}_r{round(args.real * 100)}_{synth_name}"
              f"{'' if args.synth == 'none' else f'x{args.ratio:g}'}{'_synthonly' if args.synth_only else ''}"
              f"_aug{args.aug}_{args.model}_s{args.seed}")
    if (REPO / "results" / f"{run_id}.json").exists():
        print(f"{run_id}: result exists, skipping")
        return
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    real = train_images(args.domain, args.real, args.seed)
    synth = synth_images(args.domain, synth_name, args.real, args.seed, args.ratio, len(real))
    if args.synth_only and not synth:
        raise SystemExit("--synth-only needs --synth naive|ours")
    pool = synth if args.synth_only else real + synth
    rng = random.Random(args.seed)
    epoch_list = []
    while len(epoch_list) < args.epoch_images:
        epoch_list += rng.sample(pool, len(pool))
    epoch_list = epoch_list[:args.epoch_images]

    split_dir = REPO / "data" / "splits" / args.domain
    split_files = {
        "train": epoch_list,
        **{s: [str(DATA / args.domain / "images" / f) for f in read_list(split_dir / f"{s}.txt")]
           for s in ("val", "test")}}
    for s, files in split_files.items():
        (run_dir / f"{s}.txt").write_text("\n".join(link_split(files, run_dir / "data" / s)) + "\n")
    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(run_dir), "train": "train.txt", "val": "val.txt", "test": "test.txt",
        "names": dict(enumerate(NAMES[args.domain]))}))

    from ultralytics import RTDETR, YOLO
    model = (RTDETR if args.model == "rtdetr" else YOLO)(str(WEIGHTS / MODELS[args.model]))
    aug = AUG_OFF if args.aug == "off" else {}
    t0 = time.time()
    model.train(data=str(data_yaml), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, seed=args.seed,
                deterministic=True, workers=args.workers, project=str(run_dir), name="train", exist_ok=True,
                patience=args.epochs, close_mosaic=max(1, math.ceil(args.epochs / 10)), plots=False,
                cache="ram", verbose=False, **aug)
    best = run_dir / "train" / "weights" / "best.pt"
    test = (RTDETR if args.model == "rtdetr" else YOLO)(str(best)).val(
        data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, project=str(run_dir),
        name="test", exist_ok=True, plots=False, verbose=False)

    box = test.box
    result = {
        "run_id": run_id, **vars(args), "synth_dir": synth_name,
        "n_real": len(real), "n_synth": len(synth),
        "steps": args.epochs * math.ceil(args.epoch_images / args.batch),
        "test": {"map50_95": box.map, "map50": box.map50, "precision": box.mp, "recall": box.mr,
                 "per_class": {NAMES[args.domain][c]: {"map50_95": float(box.maps[c])}
                               for c in range(len(NAMES[args.domain]))}},
        "train_seconds": round(time.time() - t0),
    }
    out = REPO / "results" / f"{run_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["test"]))


if __name__ == "__main__":
    main()
