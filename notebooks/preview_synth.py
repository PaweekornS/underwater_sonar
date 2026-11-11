"""Local preview: naive vs ours on a few train images per domain, saved as zoomed contact sheets.
Converts local_samples/<domain> colour images to intensity the same way scripts/prepare_data.py does.
Usage: python notebooks/preview_synth.py <out_dir>"""
import glob
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.bank import extract, read_boxes  # noqa: E402
from synth.compose import Options, compose  # noqa: E402
from synth.geometry import ui_text_mask  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
CFG = {"cylinder2": False, "mine": False, "aquascan": True}  # domain -> has UI overlay


for dom, has_ui in CFG.items():
    gdir = Path("local_samples/_gray") / dom
    gdir.mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted(glob.glob(f"local_samples/{dom}/images/*")):
        bgr = cv2.imread(f)
        g = gdir / (Path(f).stem + ".png")
        cv2.imwrite(str(g), bgr.max(2))
        ui = ui_text_mask(bgr) if has_ui else None
        items.append((g, f.replace("images", "labels").rsplit(".", 1)[0] + ".txt", ui))
    bank = [o for g, lab, ui in items for o in extract(g, lab, ui)]
    print(dom, "bank", len(bank), "usable", sum(o.usable for o in bank))
    rng = random.Random(0)
    tiles = []
    for k, (g, lab, ui) in enumerate(items[:4]):
        gray = cv2.imread(str(g), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        pool = [o for o in bank if o.res == f"{w}x{h}"] or bank
        scale_pool = [(o.box[2] - o.box[0], o.box[3] - o.box[1]) for o in pool]
        objs = rng.sample(pool, min(4, len(pool)))
        existing = read_boxes(lab, w, h)
        rows = []
        for method in ("naive", "ours"):
            img, labels = compose(gray, existing, objs, method, random.Random(k),
                                  Options(), scale_pool, ui)
            new = labels[len(existing):]
            full = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for _, x0, y0, x1, y1 in new:
                cv2.rectangle(full, (x0, y0), (x1, y1), (0, 0, 255), max(2, w // 400))
            crops = []
            for _, x0, y0, x1, y1 in new[:3]:
                s = max(x1 - x0, y1 - y0) * 2
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                cr = img[max(0, cy - s):cy + s, max(0, cx - s):cx + s]
                crops.append(cv2.cvtColor(cv2.resize(cr, (200, 200), interpolation=cv2.INTER_NEAREST),
                                          cv2.COLOR_GRAY2BGR))
            crops += [np.zeros((200, 200, 3), np.uint8)] * (3 - len(crops))
            tile = np.hstack([cv2.resize(full, (300, 200)), *crops])
            cv2.putText(tile, method, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            rows.append(tile)
        tiles.append(np.vstack(rows))
    cv2.imwrite(str(out / f"{dom}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
