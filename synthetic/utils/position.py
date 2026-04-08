import numpy as np
import random
import cv2

def rotate_object_and_mask(img, mask, angle=None):
    """
    Rotates the object and mask by a random angle (or specific angle).
    Expands the canvas to ensure corners are not cropped.
    """
    if angle is None:
        angle = random.uniform(0, 360)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 1. Compute the Rotation Matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2. Compute the New Bounding Dimensions (The Math part)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 3. Adjust the Matrix Translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 4. Perform Rotation: Use INTER_LINEAR for the image (smooth)
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    # Use INTER_NEAREST for the mask (sharp edges, keep it binary)
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST)

    return rotated_img, rotated_mask

def get_allowed_x_ranges(bg_width, ratio=0.3):
    left_max = int(bg_width * ratio)
    right_min = int(bg_width * (1 - ratio))
    return (0, left_max), (right_min, bg_width)

def get_foreground_extent(fg_mask):
    ys, xs = np.where(fg_mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return y0, y1, x0, x1

def sample_valid_position(bg_shape, fg_shape, side="random"):
    bg_h, bg_w = bg_shape
    fg_h, fg_w = fg_shape

    (l_min, l_max), (r_min, r_max) = get_allowed_x_ranges(bg_w)

    if side == "random":
        side = random.choice(["left", "right"])

    # Compute global allowed ranges where the foreground fits
    allowed_global_x_min = 0
    allowed_global_x_max = bg_w - fg_w
    allowed_global_y_max = bg_h - fg_h

    # Sample y safely (if fg is taller than bg, place at top)
    if allowed_global_y_max >= 0:
        y = random.randint(0, allowed_global_y_max)
    else:
        y = 0

    x = None

    # Try to sample on the requested side, but validate the range first
    if side == "left":
        cand_min = l_min
        cand_max = l_max - fg_w
        if cand_max >= cand_min and cand_min <= allowed_global_x_max:
            cand_max = min(cand_max, allowed_global_x_max)
            x = random.randint(cand_min, cand_max)
        else:
            # fallback to right side if left doesn't fit
            cand_min_r = r_min
            cand_max_r = allowed_global_x_max
            if cand_max_r >= cand_min_r:
                x = random.randint(cand_min_r, cand_max_r)
    else:
        cand_min = max(r_min, 0)
        cand_max = allowed_global_x_max
        if cand_max >= cand_min:
            x = random.randint(cand_min, cand_max)
        else:
            # fallback to left side if right doesn't fit
            cand_min_l = l_min
            cand_max_l = l_max - fg_w
            if cand_max_l >= cand_min_l and cand_min_l <= allowed_global_x_max:
                cand_max_l = min(cand_max_l, allowed_global_x_max)
                x = random.randint(cand_min_l, cand_max_l)

    # Final fallback: clamp within global allowed range or place at 0
    if x is None:
        if allowed_global_x_max >= 0:
            x = random.randint(0, allowed_global_x_max)
        else:
            x = 0

    return y, x

def bbox_from_mask(mask):
    """Compute tight bounding box from binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask, cannot compute bbox.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return x_min, y_min, x_max, y_max