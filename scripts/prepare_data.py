"""Normalize the three SSS domains into one layout and write fixed split files.

Output (not in git):  <out>/<domain>/images/<id>.png    (single-channel intensity)
                      <out>/<domain>/labels/<id>.txt    (YOLO, unified class ids)
                      <out>/<domain>/meta.csv
Output (in git):      data/splits/<domain>/{train,val,test}.txt
                      data/splits/<domain>/train_s{seed}_p{pct}.txt  (nested low-data subsets)
                      data/splits/<domain>/report.json

Usage: python scripts/prepare_data.py --raw /scratch/.../sonar-data --out /scratch/.../paweekorn/data
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.geometry import ui_text_mask  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLIT_SEED = 0
SUBSET_SEEDS = range(5)
SUBSET_PCTS = (10, 25, 50)
HASH_DIST = 6  # pHash Hamming distance at or below which two images count as near-duplicates

DOMAINS = {
    "cylinder2": {"names": ["cylinder", "manta"], "remap": {0: 0, 1: 0, 2: 1}},
    "mine": {"names": ["MILCO", "NOMBO"], "remap": {0: 0, 1: 1}},
    "aquascan": {"names": ["human"], "remap": {0: 0}},
    # Leaky reference split that mimics prior work: no dedup, downscaled copies kept. Sanity check only.
    "mine_leaky": {"names": ["MILCO", "NOMBO"], "remap": {0: 0, 1: 1}},
}


def read_yolo(path, remap):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                rows.append((remap[int(p[0])], *map(float, p[1:])))
    return rows


def collect_cylinder2(raw):
    root = raw / "cylider2"
    items = []
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        for img in sorted((root / folder / "images").glob("*.jpg")):
            survey = re.match(r"(\d{4}_\d{2}_\d{2}_exp\d)", img.name)
            items.append(dict(src=img, label=root / folder / "labels" / f"{img.stem}.txt",
                              id=img.name.split(".rf.")[0], split=split,
                              group=survey.group(1) if survey else ""))
    return items


def collect_mine(raw):
    root = raw / "mine-detection"
    return [dict(src=img, label=img.with_suffix(".txt"), id=f"{y.name}_{img.stem}", split=None, group=y.name)
            for y in sorted(root.glob("20*")) for img in sorted(y.glob("*.jpg"))]


def collect_aquascan(raw):
    root = raw / "aquaScan" / "AquaScan-1K"
    return [dict(src=img, label=root / "labels" / f"{img.stem}.txt", id=img.stem, split=None, group="")
            for img in sorted((root / "images").iterdir()) if img.suffix.lower() in {".png", ".jpg"}]


COLLECT = {"cylinder2": collect_cylinder2, "mine": collect_mine, "aquascan": collect_aquascan,
           "mine_leaky": collect_mine}


def near_dup_clusters(hashes):
    """Union-find over pairs with pHash distance <= HASH_DIST. Returns a cluster id per item."""
    bits = np.array([h.hash.flatten() for h in hashes], dtype=np.uint8)
    n = len(bits)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.nonzero(d <= HASH_DIST)[0] + i + 1:
            parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def strat_key(it):
    classes = sorted({r[0] for r in it["boxes"]})
    return f"{it['res']}|{'-'.join(map(str, classes)) or 'empty'}"


def random_split(items, clusters, fracs=(0.7, 0.15, 0.15)):
    """Stratified split where every near-duplicate cluster stays inside one split."""
    by_cluster = defaultdict(list)
    for it, c in zip(items, clusters):
        by_cluster[c].append(it)
    by_stratum = defaultdict(list)
    for members in by_cluster.values():
        by_stratum[strat_key(members[0])].append(members)
    rng = random.Random(SPLIT_SEED)
    for key in sorted(by_stratum):
        groups = by_stratum[key]
        rng.shuffle(groups)
        n = sum(len(g) for g in groups)
        cut_train, cut_val = fracs[0] * n, (fracs[0] + fracs[1]) * n
        seen = 0
        for g in groups:
            split = "train" if seen < cut_train else "val" if seen < cut_val else "test"
            for it in g:
                it["split"] = split
            seen += len(g)


def drop_train_leaks(items, clusters):
    """Official split: drop train images that are near-duplicates of a val/test image."""
    eval_clusters = {c for it, c in zip(items, clusters) if it["split"] != "train"}
    dropped = [it for it, c in zip(items, clusters) if it["split"] == "train" and c in eval_clusters]
    for it in dropped:
        it["split"] = "dropped"
    return len(dropped)


def nested_subsets(train):
    """Per seed: shuffle each stratum once and take prefixes, so 10% is inside 25% is inside 50%."""
    out = {}
    by_stratum = defaultdict(list)
    for it in train:
        by_stratum[strat_key(it)].append(it["id"])
    for seed in SUBSET_SEEDS:
        rng = random.Random(1000 + seed)
        shuffled = {k: rng.sample(sorted(v), len(v)) for k, v in sorted(by_stratum.items())}
        for pct in SUBSET_PCTS:
            ids = []
            for v in shuffled.values():
                ids += v[:max(1, round(len(v) * pct / 100))]
            out[(seed, pct)] = sorted(ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    args = ap.parse_args()

    for dom in args.domains:
        cfg = DOMAINS[dom]
        items, corrupt = [], []
        for it in COLLECT[dom](args.raw):
            try:
                with Image.open(it["src"]) as im:
                    it["w"], it["h"] = im.size
                    it["hash"] = imagehash.phash(im.convert("L"))
            except OSError:
                corrupt.append(it["src"].name)
                continue
            items.append(it)
        for it in items:
            it["res"] = f"{it['w']}x{it['h']}" if dom == "mine" else "all"
            it["boxes"] = read_yolo(it["label"], cfg["remap"])

        clusters = near_dup_clusters([it["hash"] for it in items])
        if dom == "mine_leaky":
            clusters = list(range(len(items)))
        n_downscaled = 0
        if dom == "mine":
            # 416² images that match a 1024² image are downscaled copies of it: keep only the original.
            has_big = {c for it, c in zip(items, clusters) if it["w"] == 1024}
            keep = [not (it["w"] == 416 and c in has_big) for it, c in zip(items, clusters)]
            n_downscaled = keep.count(False)
            items = [it for it, k in zip(items, keep) if k]
            clusters = [c for c, k in zip(clusters, keep) if k]
        n_dropped = drop_train_leaks(items, clusters) if dom == "cylinder2" else random_split(items, clusters) or 0

        dst = args.out / dom
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        for it in items:
            # Sonar colour is a display palette over one intensity channel; store that channel only, so
            # the model never sees palette differences between devices or software (lossless PNG).
            out_img = dst / "images" / f"{it['id']}.png"
            if not out_img.exists():
                with Image.open(it["src"]) as im:
                    rgb = np.asarray(im.convert("RGB"))
                Image.fromarray(rgb.max(2)).save(out_img)
                if dom == "aquascan":  # UI overlay text is only detectable in colour, so keep it as a mask
                    (dst / "uimask").mkdir(exist_ok=True)
                    Image.fromarray(ui_text_mask(rgb[..., ::-1]).astype(np.uint8) * 255).save(
                        dst / "uimask" / f"{it['id']}.png")
            (dst / "labels" / f"{it['id']}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in it["boxes"]))
        with open(dst / "meta.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["id", "file", "split", "w", "h", "res", "group", "n_obj", "dup_cluster"])
            for it, c in zip(items, clusters):
                wr.writerow([it["id"], f"{it['id']}.png", it["split"], it["w"], it["h"],
                             it["res"], it["group"], len(it["boxes"]), c])

        split_dir = REPO / "data" / "splits" / dom
        split_dir.mkdir(parents=True, exist_ok=True)
        for s in ("train", "val", "test"):
            ids = sorted(f"{it['id']}.png" for it in items if it["split"] == s)
            (split_dir / f"{s}.txt").write_text("\n".join(ids) + "\n")
        train = [it for it in items if it["split"] == "train"]
        ext = {it["id"]: ".png" for it in items}
        for (seed, pct), ids in nested_subsets(train).items():
            (split_dir / f"train_s{seed}_p{pct}.txt").write_text("\n".join(i + ext[i] for i in ids) + "\n")

        cluster_sizes = Counter(clusters)
        report = {
            "names": cfg["names"],
            "hash_dist": HASH_DIST,
            "corrupt_skipped": corrupt,
            "downscaled_copies_removed": n_downscaled,
            "n_images": len(items),
            "dup_clusters_gt1": sum(1 for v in cluster_sizes.values() if v > 1),
            "images_in_dup_clusters": sum(v for v in cluster_sizes.values() if v > 1),
            "train_dropped_as_leak": n_dropped,
            "splits": {s: {"images": sum(it["split"] == s for it in items),
                           "empty": sum(it["split"] == s and not it["boxes"] for it in items),
                           "boxes": dict(Counter(cfg["names"][r[0]] for it in items if it["split"] == s
                                                 for r in it["boxes"]))}
                       for s in ("train", "val", "test", "dropped")},
            "groups_per_split": {s: dict(Counter(it["group"] for it in items if it["split"] == s))
                                 for s in ("train", "val", "test")},
        }
        (split_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(dom, json.dumps(report["splits"]), "dup clusters:", report["dup_clusters_gt1"],
              "dropped:", n_dropped)


if __name__ == "__main__":
    main()
