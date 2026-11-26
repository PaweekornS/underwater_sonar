"""Swath geometry of a full-swath SSS intensity image: nadir, water column, and where objects may go.

All images in the three domains are waterfall displays with port on the left, starboard on the right,
and a dark water-column band around a centre nadir line. The band width changes row by row with
altitude, so it is found per row.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter


@dataclass
class Swath:
    nadir_x: int               # column of the nadir line
    wc_left: np.ndarray        # per row: first seafloor column going left from nadir (exclusive edge)
    wc_right: np.ndarray       # per row: first seafloor column going right from nadir
    seafloor: np.ndarray       # bool HxW: pixels where an object may be placed
    dark_level: float          # intensity of the water column / shadow level

    def range_norm(self, x, y):
        """Normalised ground range in [0, 1]: 0 at the first bottom return, 1 at the swath edge."""
        w = self.seafloor.shape[1]
        if x >= self.nadir_x:
            return float(np.clip((x - self.wc_right[y]) / max(1, w - self.wc_right[y]), 0, 1))
        return float(np.clip((self.wc_left[y] - x) / max(1, self.wc_left[y]), 0, 1))

    def side(self, x):
        """+1 for starboard (shadow points right), -1 for port (shadow points left)."""
        return 1 if x >= self.nadir_x else -1


def flat_mask(gray, win=15, std_thresh=1.0):
    """Regions with almost no local variation: UI panels, black padding, letterbox bars.
    Real seafloor always carries speckle, so flat means 'not sonar data'."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (win, win))
    var = cv2.blur(g * g, (win, win)) - mean * mean
    return np.sqrt(np.clip(var, 0, None)) < std_thresh


def ui_text_mask(bgr):
    """Bright, colourless pixels: white/grey overlay text and icons on a colour sonar palette.
    Needs the original colour screenshot; computed once in prepare_data and stored as a mask file."""
    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    text = (hsv[..., 1] < 40) & (hsv[..., 2] > 200)
    return cv2.dilate(text.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0


def analyze(gray, ui_mask=None):
    gray = gray.astype(np.float32)
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2.0, w / 200), sigmaY=max(2.0, h / 100))

    # Nadir: centre of the dark band in the middle 40% of the image.
    prof = smooth.mean(0)
    lo, hi = int(0.3 * w), int(0.7 * w)
    floor_level = np.percentile(prof, 75)
    band = prof[lo:hi] < 0.5 * floor_level
    if band.any():
        cols = np.nonzero(band)[0] + lo
        nadir_x = int((cols.min() + cols.max()) / 2)
    else:
        nadir_x = w // 2

    # Per row, walk outward from nadir until the smoothed intensity reaches half the seafloor level.
    thresh = 0.5 * floor_level
    wc_left = np.zeros(h, int)
    wc_right = np.zeros(h, int)
    for y in range(h):
        row = smooth[y]
        right = np.nonzero(row[nadir_x:] > thresh)[0]
        left = np.nonzero(row[:nadir_x][::-1] > thresh)[0]
        wc_right[y] = nadir_x + (right[0] if right.size else w - nadir_x)
        wc_left[y] = nadir_x - (left[0] if left.size else nadir_x)
    k = max(3, h // 40) | 1
    wc_right = median_filter(wc_right, k)
    wc_left = median_filter(wc_left, k)

    cols = np.arange(w)[None, :]
    outside_wc = (cols < wc_left[:, None]) | (cols >= wc_right[:, None])
    bad = flat_mask(gray)
    if ui_mask is not None:
        bad |= ui_mask
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    seafloor = outside_wc & ~bad

    wc = gray[~outside_wc & ~bad]
    dark = float(np.median(wc)) if wc.size > 100 else float(np.percentile(gray, 3))
    return Swath(nadir_x, wc_left, wc_right, seafloor, dark)
