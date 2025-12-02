import numpy as np
from scipy.stats import weibull_min
from PIL import Image
from tqdm import tqdm
import random
import os

# --- 2. Parameters Setup ---
NUM_OBJECTS_PER_BG = 1
MIN_SCALE_FACTOR = 0.05
MAX_SCALE_FACTOR = 0.15

OBJECT_COLOR = 255
SHADOW_COLOR = 0
ACTIVE_COLOR_TOLERANCE = 5  # tolerance around pure white/black in mask

# Weibull quantiles for "bright" (object) and "dark" (shadow)
OBJECT_QUANTILE = 0.5
SHADOW_QUANTILE = 0.25

# For fitting speed
MAX_PIXELS_FOR_WEIBULL = 4000

def fit_weibull_from_background(bg_np):
    """
    Fit Weibull distributions for object (bright) and shadow (dark) intensities
    using the grayscale of the background image.
    """
    # bg_np: float32, shape (H, W, 3)
    # Convert to grayscale
    gray = 0.299 * bg_np[:, :, 0] + 0.587 * bg_np[:, :, 1] + 0.114 * bg_np[:, :, 2]
    flat = gray.flatten()
    flat = flat[np.isfinite(flat)]
    flat = flat[(flat >= 0) & (flat <= 255)]

    if len(flat) == 0:
        # fallback mid-gray
        return (1.0, 0.0, 128.0), (1.0, 0.0, 64.0)

    # Subsample for speed
    if len(flat) > MAX_PIXELS_FOR_WEIBULL:
        idx = np.random.randint(0, len(flat), size=MAX_PIXELS_FOR_WEIBULL)
        flat = flat[idx]

    low_thr = np.quantile(flat, SHADOW_QUANTILE)
    high_thr = np.quantile(flat, OBJECT_QUANTILE)

    shadow_pixels = flat[flat <= low_thr]
    object_pixels = flat[flat >= high_thr]

    if len(shadow_pixels) < 500:
        shadow_pixels = flat[flat <= np.median(flat)]
    if len(object_pixels) < 500:
        object_pixels = flat[flat >= np.median(flat)]

    # Fit Weibull; floc=0 to avoid negative loc drifting
    c_sh, loc_sh, scale_sh = weibull_min.fit(shadow_pixels, floc=0)
    c_obj, loc_obj, scale_obj = weibull_min.fit(object_pixels, floc=0)

    dist_shadow = (c_sh, loc_sh, scale_sh)
    dist_object = (c_obj, loc_obj, scale_obj)
    return dist_object, dist_shadow

def sample_from_weibull(dist_params, n):
    c, loc, scale = dist_params
    if n <= 0:
        return np.array([], dtype=np.float32)
    samples = weibull_min.rvs(c, loc=loc, scale=scale, size=n)
    samples = np.clip(samples, 0, 255).astype(np.float32)
    return samples

# --- Utility functions ---
def semisynthetic(output_dir, background_dir, object_dir):
    print("--- Starting Procedural Semi-Synthetic Pipeline (Weibull-based) ---")
    os.makedirs(output_dir, exist_ok=True)

    try:
        background_files = [
            f for f in os.listdir(background_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        object_files = [
            f for f in os.listdir(object_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    except FileNotFoundError as e:
        print(f"**ERROR:** Path check failed. Please verify the directory paths: {e}")
        return

    # Load object masks (L, single-channel; values assumed 0=shadow, 255=object, mid=neutral)
    objects = []
    for obj_file in object_files:
        try:
            obj_mask = Image.open(os.path.join(object_dir, obj_file)).convert("L")
            objects.append(obj_mask)
        except Exception as e:
            print(f"!! Error loading Object Mask {obj_file}: {e}")

    if not objects:
        print("**ERROR:** Could not load any Object Masks.")
        return

    for bg_idx, bg_file in tqdm(enumerate(background_files), total=len(background_files)):
        background_path = os.path.join(background_dir, bg_file)
        composite_pil = Image.open(background_path).convert("RGB")
        bg_np = np.array(composite_pil, dtype=np.float32)
        bg_height, bg_width, _ = bg_np.shape

        # Fit Weibull for this background
        dist_object, dist_shadow = fit_weibull_from_background(bg_np)

        # Work copy for blending (NumPy)
        composite_np = bg_np.copy()

        for _ in range(NUM_OBJECTS_PER_BG):
            random_object_mask = random.choice(objects)

            # Random scale relative to background width
            scale_factor = random.uniform(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
            new_obj_width = int(bg_width * scale_factor)

            w_percent = new_obj_width / float(random_object_mask.size[0])
            new_obj_height = int(float(random_object_mask.size[1]) * w_percent)

            if new_obj_width <= 0 or new_obj_height <= 0:
                continue

            resized_mask = random_object_mask.resize(
                (new_obj_width, new_obj_height),
                Image.Resampling.NEAREST
            )

            # Random rotation with neutral fill
            fill_color = 128
            rotated_mask = resized_mask.rotate(
                random.randint(0, 359),
                expand=True,
                resample=Image.Resampling.NEAREST,
                fillcolor=fill_color
            )

            mask_np = np.array(rotated_mask, dtype=np.float32)
            obj_h, obj_w = mask_np.shape

            # --- Restrict placement to leftmost 20% or rightmost 20% ---
            band_width = int(0.2 * bg_width)

            # Left band: x in [0, band_width - obj_w]
            left_start_min = 0
            left_start_max = band_width - obj_w

            # Right band: x in [0.8*W, W - obj_w]
            right_start_min = int(0.8 * bg_width)
            right_start_max = bg_width - obj_w

            valid_bands = []
            if left_start_max >= left_start_min:
                valid_bands.append(("left", left_start_min, left_start_max))
            if right_start_max >= right_start_min:
                valid_bands.append(("right", right_start_min, right_start_max))

            if not valid_bands:
                # Cannot place object on this background size
                continue

            band_name, x_min_allowed, x_max_allowed = random.choice(valid_bands)
            random_x = random.randint(x_min_allowed, x_max_allowed)

            # y anywhere vertically
            max_y = bg_height - obj_h
            if max_y <= 0:
                continue
            random_y = random.randint(0, max_y)

            # --- Active mask for object/shadow ---

            is_object = np.logical_and(
                mask_np >= OBJECT_COLOR - ACTIVE_COLOR_TOLERANCE,
                mask_np <= OBJECT_COLOR + ACTIVE_COLOR_TOLERANCE
            )
            is_shadow = np.logical_and(
                mask_np >= SHADOW_COLOR - ACTIVE_COLOR_TOLERANCE,
                mask_np <= SHADOW_COLOR + ACTIVE_COLOR_TOLERANCE
            )
            is_active = np.logical_or(is_object, is_shadow)

            if not np.any(is_active):
                continue

            alpha_np = np.where(is_active, 255, 0).astype(np.uint8)
            alpha_channel_pil = Image.fromarray(alpha_np, mode='L')

            # --- Crop background area ---
            y1, y2 = random_y, random_y + obj_h
            x1, x2 = random_x, random_x + obj_w
            bg_area_np = composite_np[y1:y2, x1:x2, :].copy()

            # --- Sample intensities from Weibull for object & shadow ---
            num_obj_pix = int(is_object.sum())
            num_sh_pix = int(is_shadow.sum())

            obj_samples = sample_from_weibull(dist_object, num_obj_pix)
            sh_samples = sample_from_weibull(dist_shadow, num_sh_pix)

            # Create 2D intensity maps
            obj_intensity = np.zeros_like(mask_np, dtype=np.float32)
            sh_intensity = np.zeros_like(mask_np, dtype=np.float32)

            if num_obj_pix > 0:
                obj_intensity[is_object] = obj_samples
            if num_sh_pix > 0:
                sh_intensity[is_shadow] = sh_samples

            # --- Apply intensities to RGB channels (SSS is effectively grayscale) ---
            blended_area_np = bg_area_np.copy()
            for c in range(3):  # R,G,B
                ch = blended_area_np[:, :, c]
                # object pixels
                ch = np.where(is_object, obj_intensity, ch)
                # shadow pixels
                ch = np.where(is_shadow, sh_intensity, ch)
                blended_area_np[:, :, c] = ch

            # Convert blended patch to PIL and paste with alpha
            blended_object_rgb_pil = Image.fromarray(
                np.uint8(np.clip(blended_area_np, 0, 255)),
                mode='RGB'
            )
            blended_object_rgba_pil = Image.merge(
                'RGBA',
                blended_object_rgb_pil.split() + (alpha_channel_pil,)
            )

            composite_pil.paste(
                blended_object_rgba_pil,
                (random_x, random_y),
                mask=blended_object_rgba_pil
            )

        # Save result
        output_path = os.path.join(output_dir, f"semisynth-{bg_idx:04d}.png")
        composite_pil.save(output_path, quality=95)
        # print(f"Saved: {output_path}")

    print("\nProcessing complete!")
