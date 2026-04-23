import numpy as np
import random
import cv2
import os

def roi_thresholding(segment_path):
    segment = cv2.imread(segment_path, 0)
    segment[ segment >= 200 ] = 255
    segment[ segment <= 50 ] = 0
    return segment

def filter_object(segment_path):
    segment = roi_thresholding(segment_path)
    img_path = segment_path.replace("masked", "resized")
    img = cv2.imread(img_path, 0)
    
    obj_canvas = np.zeros(shape=segment.shape)
    obj_canvas[ segment == 255 ] = img[ segment == 255 ]
    
    # --- Random square resize (32 to 128) ---
    class_name = os.path.basename(segment_path)[:-4]
    
    if "ship" in class_name:
        size = random.randint(96, 128)
    elif "plane" in class_name:
        size = random.randint(64, 96)
    else:
        size = random.randint(64, 128)
        
    obj_canvas = cv2.resize(obj_canvas, (size, size), interpolation=cv2.INTER_LINEAR)
    return obj_canvas

def sample_valid_position(bg_shape, fg_shape, side="random", threshold=0.2):
    bg_h, bg_w = bg_shape
    fg_h, fg_w = fg_shape

    # Random side selection
    if side == "random":
        side = random.choice(["left", "right"])

    # --- Y (vertical placement) ---
    max_y = bg_h - fg_h
    y = random.randint(0, max_y) if max_y >= 0 else 0

    # --- X (strict region constraint) ---
    max_x_global = bg_w - fg_w

    if max_x_global < 0:
        # foreground wider than background → force 0
        return y, 0

    if side == "left":
        x_min = 0
        x_max = int(threshold * bg_w) - fg_w

    else:  # right
        x_min = int((1-threshold) * bg_w)
        x_max = max_x_global

    # --- Clamp to valid global bounds ---
    x_min = max(0, x_min)
    x_max = min(x_max, max_x_global)

    # --- Strict validity check ---
    if x_max < x_min:
        if side == "left":
            x = 0
        else:
            x = max_x_global
    else:
        x = random.randint(x_min, x_max)

    return y, x

def rotate_object(img):
    angle = random.uniform(0, 360)

    h, w = img.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return rotated

def construct_shadow(fg_img, direction="right", max_length_ratio=1.5):
    """
    Generate physically plausible sonar shadow.

    - No accumulation explosion
    - Smooth decay
    - Direction-aware
    """

    mask = (fg_img > 0).astype(np.uint8)

    h, w = mask.shape
    shadow = np.zeros_like(mask, dtype=np.float32)

    max_shift = int(w * max_length_ratio)
    max_shift = min(max_shift, 120)  # clamp to avoid long tails

    for i in range(1, max_shift):
        # Exponential decay (controls shadow fading)
        decay = np.exp(-i / 30.0)

        if direction == "right":
            shifted = np.roll(mask, i, axis=1)
            shifted[:, :i] = 0
        else:
            shifted = np.roll(mask, -i, axis=1)
            shifted[:, -i:] = 0

        # IMPORTANT: use max instead of sum
        shadow = np.maximum(shadow, shifted * decay)

    # Normalize safely (no amplification of noise)
    if shadow.max() > 0:
        shadow = shadow / shadow.max()

    # Convert to usable intensity map
    shadow = (shadow * 255).astype(np.uint8)

    # --- Smooth edges (remove jagged artifacts) ---
    shadow = cv2.GaussianBlur(shadow, (9, 9), 0)

    # --- Remove weak noise ---
    shadow[shadow < 10] = 0

    return shadow
