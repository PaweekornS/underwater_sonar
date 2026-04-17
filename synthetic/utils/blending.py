import numpy as np
import cv2

def match_local_stats(bg_img, fg_img, fg_mask, top, left):
    h, w = fg_mask.shape
    bg_h, bg_w = bg_img.shape

    roi = bg_img[top:top+h, left:left+w]

    if roi.size == 0: return fg_img

    # Local Estimation (The Math)
    bg_floor = np.percentile(roi, 5)   # The darkest valid shadow
    bg_ceil = np.percentile(roi, 95)   # The brightest

    valid_fg_pixels = fg_img[fg_mask > 0]

    if valid_fg_pixels.size == 0: return fg_img

    fg_min = valid_fg_pixels.min()
    fg_max = valid_fg_pixels.max()

    # Compute Linear Scaling Factors
    range_fg = max(fg_max - fg_min, 1e-5)
    range_bg = bg_ceil - bg_floor

    scale = range_bg / range_fg
    offset = bg_floor - (fg_min * scale)

    # convert to float for precision, then back to uint8
    fg_adjusted = fg_img.astype(np.float32) * scale + offset

    # Clip values to ensure they stay within valid image range (0-255)
    fg_adjusted = np.clip(fg_adjusted, 0, 255).astype(np.uint8)

    return fg_adjusted

def composite_foreground(bg_img, fg_img, fg_mask, top, left):
    """Composite fg_img onto bg_img using fg_mask."""
    out = bg_img.copy()

    h, w = fg_mask.shape
    roi = out[top:top+h, left:left+w]

    # Only overwrite where mask == 1
    roi[fg_mask > 0] = fg_img[fg_mask > 0]

    out[top:top+h, left:left+w] = roi

    return out