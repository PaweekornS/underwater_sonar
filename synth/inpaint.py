"""Box-conditioned diffusion inpainting for SSS: shared geometry for training crops and generation.

A training example is a square context window around a real object, resized to RES, with a mask over the
object's box extended towards far range so the cast shadow is inside the mask. Every window is stored in
the canonical orientation (shadow pointing right, i.e. starboard); port-side windows are mirrored.
Generation uses the same geometry, so the model always draws an object whose shadow points right, and the
result is mirrored back for the port side.
"""
import cv2
import numpy as np

RES = 512
CONTEXT = 3.0        # window side = CONTEXT * max(box w, h)
MIN_WINDOW = 64      # px in the source image
SHADOW_EXT = 1.0     # mask extends this many box widths towards far range


def prompt(domain, cls_name):
    return f"side-scan sonar image, {domain}, {cls_name}"


def window(box, img_w, img_h):
    """Square context window (x0, y0, side) around a box, clipped to fit inside the image."""
    x0, y0, x1, y1 = box
    side = int(max(MIN_WINDOW, CONTEXT * max(x1 - x0, y1 - y0)))
    side = min(side, img_w, img_h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    wx = int(np.clip(cx - side / 2, 0, img_w - side))
    wy = int(np.clip(cy - side / 2, 0, img_h - side))
    return wx, wy, side


def mask_in_window(box, win, starboard):
    """Mask (RES x RES, float 0/1) over the box plus its far-range shadow side, in canonical orientation."""
    x0, y0, x1, y1 = box
    wx, wy, side = win
    bw = x1 - x0
    pad = 0.15 * max(bw, y1 - y0)
    mx0, mx1 = x0 - pad, x1 + pad
    if starboard:
        mx1 += SHADOW_EXT * bw
    else:
        mx0 -= SHADOW_EXT * bw
    s = RES / side
    m = np.zeros((RES, RES), np.float32)
    ys, ye = int(max(0, (y0 - pad - wy) * s)), int(min(RES, (y1 + pad - wy) * s))
    xs, xe = int(max(0, (mx0 - wx) * s)), int(min(RES, (mx1 - wx) * s))
    m[ys:ye, xs:xe] = 1
    return m if starboard else m[:, ::-1].copy()


def crop(gray, win, starboard):
    """Window as a RES x RES uint8 image in canonical orientation."""
    wx, wy, side = win
    c = cv2.resize(gray[wy:wy + side, wx:wx + side], (RES, RES), interpolation=cv2.INTER_LINEAR)
    return c if starboard else c[:, ::-1].copy()


def uncrop(gray, generated, mask, win, starboard):
    """Write the generated window back into the image, only inside a feathered mask."""
    wx, wy, side = win
    if not starboard:
        generated, mask = generated[:, ::-1], mask[:, ::-1]
    g = cv2.resize(generated.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(cv2.resize(mask, (side, side), interpolation=cv2.INTER_LINEAR), (0, 0), max(1.0, side / 128))
    roi = gray[wy:wy + side, wx:wx + side].astype(np.float32)
    gray[wy:wy + side, wx:wx + side] = np.clip(a * g + (1 - a) * roi, 0, 255).astype(np.uint8)


def signal_to_noise(gray, box):
    """How far the box deviates from the surrounding seabed, in units of seabed noise (same metric as the
    diagnostics): used to reject generations where no object appeared."""
    x0, y0, x1, y1 = box
    h, w = gray.shape
    m = max(x1 - x0, y1 - y0)
    g = gray.astype(np.float32)
    inner = g[y0:y1, x0:x1]
    ring = g[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)]
    bg, noise = np.median(ring), ring.std() + 1e-3
    return float(max(np.percentile(inner, 95) - bg, bg - np.percentile(inner, 5)) / noise)
