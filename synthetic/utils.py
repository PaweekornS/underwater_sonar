import numpy as np
import math
import random

from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter
import cv2

# -------------------------------
# Step 1: Adaptive threshold (3-class)
# -------------------------------
def adaptive_3class_segmentation(ref_img, k=31, t_lo=-1.0, t_hi=1.25):
    ref = np.array(ref_img.convert("L"), dtype=np.float32) / 255.0
    # local mean / std via box filter (approx with Gaussian here)
    m = gaussian_filter(ref, k/6)
    s = np.sqrt(np.maximum(gaussian_filter(ref**2, k/6) - m*m, 1e-6))
    Z = (ref - m) / (s + 1e-6)
    shadow = (Z < t_lo).astype(np.uint8)
    obj    = (Z > t_hi).astype(np.uint8)
    bg     = 1 - np.clip(shadow + obj, 0, 1)
    return shadow, obj, bg, ref

# -------------------------------
# Step 2: Random bg crop
# -------------------------------
def random_bg_crop(bg_mask, ref, crop_size=(128,128)):
    H, W = bg_mask.shape
    ys, xs = np.where(bg_mask > 0)
    idx = random.randrange(len(xs))
    cy, cx = ys[idx], xs[idx]
    ch, cw = crop_size
    y0 = np.clip(cy - ch//2, 0, H-ch)
    x0 = np.clip(cx - cw//2, 0, W-cw)
    patch = ref[y0:y0+ch, x0:x0+cw]
    return patch

# -------------------------------
# Step 3+4: Place mask + shadow
# -------------------------------
def _np_gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: 
        return a
    # use PIL for speed & no SciPy dependency
    img = Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.array(img, dtype=np.float32)/255.0

def place_mask_with_shadow(
    bg_patch: np.ndarray,
    mask_img: Image.Image,
    angle_deg: float = 200.0,
    scale_range=(0.25, 0.45),     # object width as fraction of bg width
    rot_range=(-20.0, 20.0),      # small rotation range
    shadow_scale: float = 1.2,    # shadow length ≈ 1.2× object size (not too big)
    shadow_blur_sigma: float = 4, # soft penumbra
    shadow_darkness: float = 0.6, # darken amount under shadow (0..1)
    highlight_gain: float = 2.0,  # brighten object a bit
):
    """
    bg_patch: float32 array in [0,1], HxW
    mask_img: PIL image (white object on black)
    returns: composite float32 HxW
    """
    bg = np.clip(bg_patch.astype(np.float32), 0, 1)
    H, W = bg.shape

    # 1) Prepare mask: scale, rotate
    #    - pick target width ~ 25–45% of bg width (can tweak)
    target_w = int(W * random.uniform(*scale_range))
    # preserve aspect
    ratio = target_w / mask_img.width
    target_h = max(1, int(mask_img.height * ratio))
    obj = mask_img.convert("L").resize((target_w, target_h), Image.BILINEAR)

    angle = random.uniform(*rot_range)
    obj = obj.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=0)

    obj_np_small = (np.array(obj, dtype=np.float32) / 255.0)
    # binarize a bit to keep edges soft but remove haze
    obj_np_small = (obj_np_small > 0.5).astype(np.float32)

    # 2) Paste onto full-size canvas at a random location
    mask_canvas = np.zeros((H, W), dtype=np.float32)
    h, w = obj_np_small.shape
    # choose top-left so the object fully fits
    max_oy = max(0, H - h)
    max_ox = max(0, W - w)
    oy = random.randint(0, max_oy) if max_oy > 0 else 0
    ox = random.randint(0, max_ox) if max_ox > 0 else 0
    mask_canvas[oy:oy+h, ox:ox+w] = np.maximum(mask_canvas[oy:oy+h, ox:ox+w], obj_np_small)

    # 3) Simple highlight on the object
    comp = bg*(1.0 - mask_canvas) + mask_canvas*np.clip(bg*highlight_gain, 0, 1)

    # 4) Build a small shadow (projected, short, soft)
    ys, xs = np.where(mask_canvas > 0.5)
    if xs.size > 0:
        obj_w = xs.max() - xs.min() + 1
        obj_h = ys.max() - ys.min() + 1
        L = int(shadow_scale * max(obj_w, obj_h))  # short-ish
        theta = math.radians(angle_deg)
        dx = int(round(math.cos(theta) * L))
        dy = int(round(math.sin(theta) * L))

        # draw a projected swath by stepping from object towards (dx,dy)
        shadow = np.zeros_like(mask_canvas, dtype=np.float32)
        steps = max(abs(dx), abs(dy), 1)
        for t in range(steps+1):
            yi = np.clip(ys + int(round(dy * t/steps)), 0, H-1)
            xi = np.clip(xs + int(round(dx * t/steps)), 0, W-1)
            shadow[yi, xi] = 1.0

        # soften penumbra and normalize
        shadow = _np_gaussian_blur(shadow, shadow_blur_sigma)
        if shadow.max() > 0:
            shadow = shadow / shadow.max()

        # do NOT darken the object body itself
        inv_obj = 1.0 - mask_canvas
        comp = comp * (1.0 - (shadow * inv_obj) * shadow_darkness)

    return np.clip(comp, 0, 1).astype(np.float32)


# -------------------------------
# Step 5: Weibull speckle
# -------------------------------
def fit_weibull(data, max_iters=20):
    x = data.flatten(); x = x[x>0]
    k = 1.5
    for _ in range(max_iters):
        xk = x**k
        A = (xk*np.log(x)).sum()/xk.sum()
        B = np.log(x).mean()
        g = 1/k + B - A
        k -= g/( -1/k**2 - ( (xk*(np.log(x)**2)).sum()*xk.sum() - (xk*np.log(x)).sum()**2 )/(xk.sum()**2) )
        k = max(k, 0.5)
    lam = (x**k).mean()**(1/k)
    return k, lam

def apply_weibull_speckle(img, k, lam):
    H,W = img.shape
    U = np.random.rand(H,W)
    speckle = lam * (-np.log(1-U+1e-8))**(1/k)
    speckle /= speckle.mean()
    return np.clip(img * speckle, 0, 1)

# -------------------------------
# Full pipeline
# -------------------------------
def semi_synthetic(mask_path, ref_path, out_path):
    ref_img = Image.open(ref_path)
    mask_img = Image.open(mask_path)

    shadow, obj, bg, ref = adaptive_3class_segmentation(ref_img)
    patch = random_bg_crop(bg, ref, crop_size=(128,128))
    comp  = place_mask_with_shadow(patch, mask_img)
    k, lam = fit_weibull(patch*255.0)
    final = apply_weibull_speckle(comp, k, lam)

    img = (final*255).astype(np.uint8)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}, Weibull k={k:.2f}, λ={lam:.2f}")
    
    return img
