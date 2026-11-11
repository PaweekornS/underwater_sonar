"""Object bank: acoustic signatures cut from the train subset of one domain (single-channel intensity).

A signature is the object's highlight (brighter than the seabed) plus its real cast shadow (darker),
taken from a crop that extends past the labelled box on the far-range side so the shadow is included
even when the label excludes it. Every entry is stored with the shadow pointing right (+range).
`naive` uses the raw labelled box instead.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from synth.geometry import analyze, flat_mask


@dataclass
class BankObject:
    cls: int
    patch: np.ndarray       # float32 crop, shadow pointing right
    bright: np.ndarray      # float32 [0,1] highlight mask
    dark: np.ndarray        # float32 [0,1] cast-shadow mask
    box: tuple              # labelled box inside the crop (x0, y0, x1, y1), same orientation
    stretch_from: int       # crop column where the shadow starts; columns right of it scale with range
    range_norm: float       # normalised ground range of the source position
    raw_box: np.ndarray     # uint8 labelled box as in the source image (naive copy-paste)
    ring_mean: float        # mean seabed level around the source object
    res: str                # image-size group, e.g. "1024x1024"
    usable: bool            # has a detectable signature


def read_boxes(label_path, w, h):
    boxes = []
    for line in open(label_path):
        p = line.split()
        if len(p) == 5:
            c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
            boxes.append((c, int(round((x - bw / 2) * w)), int(round((y - bh / 2) * h)),
                          int(round((x + bw / 2) * w)), int(round((y + bh / 2) * h))))
    return boxes


def _keep_touching(mask, seed):
    """Components of `mask` that touch `seed` (both uint8)."""
    n, lab = cv2.connectedComponents(mask)
    ids = np.unique(lab[(seed > 0) & (mask > 0)])
    return np.isin(lab, ids[ids > 0]).astype(np.uint8)


def extract(image_path, label_path, ui_mask=None):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    swath = analyze(gray, ui_mask)
    g = gray.astype(np.float32)
    boxes = read_boxes(label_path, w, h)
    occupied = np.zeros((h, w), bool)
    for _, x0, y0, x1, y1 in boxes:
        occupied[max(0, y0):y1, max(0, x0):x1] = True

    out = []
    for c, x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        bw, bh = x1 - x0, y1 - y0
        if bw < 3 or bh < 3:
            continue
        side = swath.side((x0 + x1) / 2)
        near, far = int(0.3 * bw) + 2, int(2.0 * bw) + 2
        cx0, cx1 = (max(0, x0 - near), min(w, x1 + far)) if side > 0 else (max(0, x0 - far), min(w, x1 + near))
        cy0, cy1 = max(0, y0 - int(0.3 * bh) - 2), min(h, y1 + int(0.3 * bh) + 2)

        m = max(bw, bh)
        ring = np.zeros((h, w), bool)
        ring[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m] = True
        ring[cy0:cy1, cx0:cx1] = False
        ring &= ~occupied & swath.seafloor
        if ring.sum() < 20:
            continue
        rm, rs = float(g[ring].mean()), float(g[ring].std()) + 1e-3

        crop = g[cy0:cy1, cx0:cx1]
        sm = cv2.GaussianBlur(crop, (0, 0), 1.0)
        in_box = np.zeros(crop.shape, np.uint8)
        in_box[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = 1
        bright = ((sm > rm + 1.5 * rs) & (in_box > 0)).astype(np.uint8)
        # A shadow still carries noise; flat black is a data gap or padding, not a shadow.
        dark = ((sm < rm - 1.0 * rs) & (sm < 0.7 * rm) & ~flat_mask(crop, win=5, std_thresh=0.5)).astype(np.uint8)
        dark[: y0 - cy0] = 0
        dark[y1 - cy0:] = 0
        dark = _keep_touching(dark, cv2.dilate(np.maximum(bright, in_box), np.ones((3, 3), np.uint8)))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        box = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
        if side < 0:  # store with the shadow pointing right
            crop, bright, dark = crop[:, ::-1], bright[:, ::-1], dark[:, ::-1]
            cw = crop.shape[1]
            box = (cw - box[2], box[1], cw - box[0], box[3])
        cols = np.nonzero(bright.any(0))[0]
        stretch_from = int(cols.max() + 1) if cols.size else (box[0] + box[2]) // 2
        feather = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.8)
        out.append(BankObject(
            cls=c, patch=np.ascontiguousarray(crop), bright=feather(bright), dark=feather(dark), box=box,
            stretch_from=stretch_from, range_norm=swath.range_norm((x0 + x1) // 2, (y0 + y1) // 2),
            raw_box=gray[y0:y1, x0:x1].copy(), ring_mean=rm, res=f"{w}x{h}",
            usable=bright.sum() + dark.sum() >= 6))
    return out
