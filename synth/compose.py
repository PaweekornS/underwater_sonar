"""Paste bank objects onto a single-channel intensity background.

`ours` = physics-aware: seafloor-only placement, train-distribution scale, the object's real acoustic
signature with its shadow pointing away from nadir and its length rescaled to the target range,
background-matched feathered blending, speckle matching. Each component can be switched off for the
leave-one-out ablation. `naive` = raw labelled box, original size, uniform random position, hard paste.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from synth.geometry import analyze

COMPONENTS = ("scale", "placement", "blending", "speckle", "shadow")


@dataclass
class Options:
    off: set = field(default_factory=set)   # components disabled for ablation
    blend: str = "feather"                  # "feather" (main) or "poisson" (ablation)


def shadow_factor(r):
    """Relative shadow length at normalised ground range r (flat seabed, constant altitude: the shadow of
    an object of fixed height grows with range)."""
    return 0.3 + 2.5 * r


def _free(occupied, x0, y0, x1, y1):
    h, w = occupied.shape
    return 0 <= x0 and 0 <= y0 and x1 <= w and y1 <= h and not occupied[y0:y1, x0:x1].any()


def _mark(occupied, x0, y0, x1, y1, pad=4):
    occupied[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True


def paste_naive(img, objects, occupied, rng):
    h, w = img.shape
    labels = []
    for obj in objects:
        bh, bw = obj.raw_box.shape
        for _ in range(50):
            x, y = rng.randint(0, w - bw), rng.randint(0, h - bh)
            if _free(occupied, x, y, x + bw, y + bh):
                img[y:y + bh, x:x + bw] = obj.raw_box
                _mark(occupied, x, y, x + bw, y + bh)
                labels.append((obj.cls, x, y, x + bw, y + bh))
                break
    return labels


def speckle(patch, alpha, target_cv, rng_np):
    """Multiplicative Gamma speckle restoring the texture that resizing smoothed away. The loss is measured
    on the seabed pixels of the crop (outside the signature), which went through the same resize, and
    only the missing part of the variance is added."""
    m = alpha < 0.05
    if m.sum() < 10:
        return patch
    cur = patch[m].std() / (patch[m].mean() + 1e-6)
    add = target_cv ** 2 - cur ** 2
    if add <= 1e-4:
        return patch
    looks = 1.0 / add
    return patch * rng_np.gamma(looks, 1.0 / looks, size=patch.shape).astype(np.float32)


def _stretch(arrs, col, k):
    """Resample columns right of `col` by factor k (the shadow part), leaving the object untouched."""
    out = []
    for a in arrs:
        right = a[:, col:]
        nw = max(1, int(round(right.shape[1] * k)))
        out.append(np.hstack([a[:, :col], cv2.resize(right, (nw, a.shape[0]), interpolation=cv2.INTER_LINEAR)])
                   if right.shape[1] else a)
    return out


def _signature(obj, s, k, use_shadow):
    """Scaled, range-stretched signature with the shadow pointing right. Returns arrays and label box."""
    patch, bright, dark = obj.patch, obj.bright, obj.dark if use_shadow else np.zeros_like(obj.dark)
    x0, y0, x1, y1 = obj.box
    col = obj.stretch_from
    if s != 1.0:
        size = (max(3, round(patch.shape[1] * s)), max(3, round(patch.shape[0] * s)))
        patch, bright, dark = (cv2.resize(a, size, interpolation=cv2.INTER_LINEAR) for a in (patch, bright, dark))
        x0, y0, x1, y1, col = (int(round(v * s)) for v in (x0, y0, x1, y1, col))
    if use_shadow and k != 1.0:
        patch, bright, dark = _stretch((patch, bright, dark), col, k)
        x1 = x1 if x1 <= col else col + int(round((x1 - col) * k))
    return patch, bright, dark, (x0, y0, x1, y1)


def paste_ours(level, objects, occupied, rng, opt, scale_pool, ui_mask=None):
    rng_np = np.random.default_rng(rng.randrange(2 ** 32))
    h, w = level.shape
    swath = analyze(level, ui_mask)
    use_shadow = "shadow" not in opt.off
    labels = []
    for obj in objects:
        bw0, bh0 = obj.box[2] - obj.box[0], obj.box[3] - obj.box[1]
        s = 1.0
        if "scale" not in opt.off and scale_pool:
            tw, th = rng.choice(scale_pool)
            s = float(np.sqrt((tw * th) / (bw0 * bh0)))

        placed = None
        for _ in range(100):
            bw, bh = max(3, round(bw0 * s)), max(3, round(bh0 * s))
            if "placement" in opt.off:
                bx, by = rng.randint(0, w - bw), rng.randint(0, h - bh)
            else:
                ys, xs = np.nonzero(swath.seafloor[: h - bh, : w - bw])
                if len(xs) == 0:
                    break
                i = rng.randrange(len(xs))
                bx, by = int(xs[i]), int(ys[i])
                if not swath.seafloor[by:by + bh, bx:bx + bw].all():
                    continue
            side = swath.side(bx + bw / 2)
            r = swath.range_norm(bx + bw // 2, by + bh // 2)
            k = float(np.clip(shadow_factor(r) / shadow_factor(obj.range_norm), 0.4, 3.0))
            patch, bright, dark, (lx0, ly0, lx1, ly1) = _signature(obj, s, k, use_shadow)
            ph, pw = patch.shape
            if side < 0:  # shadow must point left on the port side
                patch, bright, dark = patch[:, ::-1], bright[:, ::-1], dark[:, ::-1]
                lx0, lx1 = pw - lx1, pw - lx0
            ox, oy = bx - lx0, by - ly0  # crop origin so that the label box lands on (bx, by)
            if _free(occupied, ox, oy, ox + pw, oy + ph):
                placed = (patch, bright, dark, ox, oy, lx0, ly0, lx1, ly1)
                break
        if placed is None:
            continue
        patch, bright, dark, ox, oy, lx0, ly0, lx1, ly1 = placed
        ph, pw = patch.shape

        roi = level[oy:oy + ph, ox:ox + pw]
        local = level[max(0, oy - ph):oy + 2 * ph, max(0, ox - pw):ox + 2 * pw]
        patch = patch.astype(np.float32)
        if "blending" not in opt.off:
            # Echo strength scales with the receiver gain/TVG of the target area: match the seabed level
            # by ratio, which keeps the object's contrast relative to its surroundings.
            patch = patch * (local.mean() / max(obj.ring_mean, 1.0))
        alpha = np.maximum(bright, dark)
        if "speckle" not in opt.off:
            patch = speckle(np.clip(patch, 1, 255), alpha, local.std() / (local.mean() + 1e-6), rng_np)
        patch = np.clip(patch, 0, 255)

        if "blending" in opt.off:
            level[oy:oy + ph, ox:ox + pw] = np.where(alpha > 0.3, patch, roi)
        elif opt.blend == "poisson" and ph > 4 and pw > 4:
            src = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # seamlessClone needs 3 channels
            dst = cv2.cvtColor(np.clip(level, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            mask = (alpha > 0.3).astype(np.uint8) * 255
            if mask.any():
                level[:] = cv2.seamlessClone(src, dst, mask, (ox + pw // 2, oy + ph // 2), cv2.NORMAL_CLONE)[..., 0]
        else:
            level[oy:oy + ph, ox:ox + pw] = alpha * patch + (1 - alpha) * roi

        _mark(occupied, ox, oy, ox + pw, oy + ph)
        labels.append((obj.cls, ox + lx0, oy + ly0, ox + lx1, oy + ly1))
    return labels


def compose(gray, existing_boxes, objects, method, rng, opt=None, scale_pool=None, ui_mask=None):
    """gray: uint8 HxW. Returns (image, labels). Existing labelled objects are kept and never overlapped."""
    h, w = gray.shape
    occupied = np.zeros((h, w), bool)
    for _, x0, y0, x1, y1 in existing_boxes:
        _mark(occupied, x0, y0, x1, y1, pad=8)
    if method == "naive":
        img = gray.copy()
        return img, list(existing_boxes) + paste_naive(img, objects, occupied, rng)
    level = gray.astype(np.float32)
    new = paste_ours(level, [o for o in objects if o.usable], occupied, rng, opt or Options(), scale_pool, ui_mask)
    return np.clip(np.round(level), 0, 255).astype(np.uint8), list(existing_boxes) + new
