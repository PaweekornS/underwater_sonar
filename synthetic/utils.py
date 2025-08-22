import numpy as np
import cv2
from skimage.filters import threshold_multiotsu, gaussian
from scipy.stats import weibull_min


def ensure_uint8(img):
    if img.dtype == np.uint8: return img
    img = img.astype(np.float32)
    img -= img.min()
    m = img.max()
    if m > 0: img /= m
    return (img * 255).astype(np.uint8)


def translate_mask(mask, dx, dy, scale=1.0, blur_sigma=1.2):
    mask = (mask>0).astype(np.uint8)*255
    h, w = mask.shape
    if scale != 1.0:
        mask = cv2.resize(mask, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
        h, w = mask.shape
    M = np.float32([[1,0,dx],[0,1,dy]])
    shifted = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if blur_sigma and blur_sigma > 0:
        shifted = ensure_uint8(gaussian(shifted/255.0, blur_sigma)*255)
    return (shifted>0).astype(np.uint8)*255


def segment_reference_multiotsu(ref_gray, classes=3):
    ref = ensure_uint8(ref_gray)
    thr = threshold_multiotsu(ref, classes=classes)
    regions = np.digitize(ref, bins=thr)
    
    # 0=shadow (darkest), 1=background, 2=highlight (brightest)
    shadow_mask = (regions==0).astype(np.uint8)*255
    bg_mask     = (regions==1).astype(np.uint8)*255
    hl_mask     = (regions==2).astype(np.uint8)*255
    return hl_mask, shadow_mask, bg_mask


def fit_weibull_from_region(img_u8, region_mask):
    vals = img_u8[region_mask>0].astype(np.float64)
    if vals.size < 50:
        return dict(c=1.5, scale=max(1.0, vals.std() if vals.size>0 else 10.0), min_val=float(vals.min() if vals.size>0 else 0.0))
    min_val = vals.min()
    shifted = np.clip(vals - min_val, 0, None) + 1e-6
    c, loc, scale = weibull_min.fit(shifted, floc=0)
    return dict(c=float(c), scale=float(scale), min_val=float(min_val))


def sample_weibull(params, n):
    c, scale, min_val = params["c"], params["scale"], params["min_val"]
    samples = weibull_min.rvs(c, loc=0, scale=scale, size=n) + min_val
    return np.clip(samples, 0, 255).astype(np.uint8)


def build_background_from_reference(ref_gray, bg_mask, out_shape, tile_size=128):
    ref = ensure_uint8(ref_gray)
    H, W = out_shape
    canvas = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(bg_mask>0)
    if len(ys) == 0:
        return cv2.resize(ref, (W, H), interpolation=cv2.INTER_LINEAR)
    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            idx = np.random.randint(0, len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            y1 = max(0, cy - tile_size//2); y2 = y1 + tile_size
            x1 = max(0, cx - tile_size//2); x2 = x1 + tile_size
            patch = ref[y1:y2, x1:x2]
            if patch.shape[:2] != (tile_size, tile_size):
                patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            canvas[y0:y0+tile_size, x0:x0+tile_size] = patch[:min(tile_size, H-y0), :min(tile_size, W-x0)]
    return canvas


def feather_blend(dst, src, mask, feather_sigma=2.5):
    mask_f = gaussian((mask>0).astype(np.float32), sigma=feather_sigma)
    mask_f = np.clip(mask_f, 0, 1)
    return ensure_uint8(dst*(1-mask_f) + src*mask_f)


def generate_semisynthetic_from_mask(
    mask_path, reference_path,
    out_w=512, out_h=256,
    obj_height_frac=0.6,   # object target height as frac of output height
    x_offset_frac=0.15,    # left placement (0..1)
    shadow_dx=35, shadow_dy=15, shadow_scale=1.0,
    downsample_azimuth=0
):
    # read binary mask (255=object)
    mask = cv2.imread(mask_path, 0)
    mask = cv2.bitwise_not(mask)   # switch black and white
    mask = (mask>0).astype(np.uint8)*255

    ref = cv2.imread(reference_path, 0)

    H, W = out_h, out_w
    ys, xs = np.where(mask>0)
    if len(ys)==0: raise ValueError("Mask has no foreground (all zeros).")
    obj_h = ys.max()-ys.min()+1; obj_w = xs.max()-xs.min()+1

    # scale to desired fraction of canvas height
    target_h = max(1, int(obj_height_frac * H))
    s = min(target_h/obj_h, 0.9*W/obj_w)
    mask_r = cv2.resize(mask, (int(mask.shape[1]*s), int(mask.shape[0]*s)), interpolation=cv2.INTER_NEAREST)

    # place on canvas
    canvas_obj = np.zeros((H, W), dtype=np.uint8)
    y0 = (H - mask_r.shape[0])//2
    x0 = int(x_offset_frac * W)
    canvas_obj[y0:y0+mask_r.shape[0], x0:x0+mask_r.shape[1]] = mask_r

    # shadow from translated object area
    shadow_mask = translate_mask(canvas_obj, shadow_dx, shadow_dy, scale=shadow_scale, blur_sigma=1.2)
    shadow_mask[canvas_obj>0] = 0  # avoid overlap dominance

    # reference segmentation & Weibull fit
    hl_ref, sh_ref, bg_ref = segment_reference_multiotsu(ref, classes=3)
    ref_u8 = ensure_uint8(ref)
    hl_params = fit_weibull_from_region(ref_u8, hl_ref)
    sh_params = fit_weibull_from_region(ref_u8, sh_ref)

    # background mosaic
    bg = build_background_from_reference(ref_u8, bg_ref, out_shape=(H, W), tile_size=128)

    # paint object
    out = bg.copy()
    n_obj = int((canvas_obj>0).sum())
    if n_obj:
        obj_vals = sample_weibull(hl_params, n_obj)
        obj_vals = np.clip(obj_vals * 1.2, 0, 255)  # 20% brighter
        out[canvas_obj > 0] = obj_vals


    # paint shadow (feathered)
    n_sh = int((shadow_mask>0).sum())
    if n_sh:
        shadow_layer = out.copy()
        shadow_layer[shadow_mask>0] = sample_weibull(sh_params, n_sh)
        out = feather_blend(out, shadow_layer, shadow_mask, feather_sigma=2.5)

    # mild blur for acoustic smoothing
    out = ensure_uint8(cv2.GaussianBlur(out, (0,0), 0.8))

    # optional azimuth downsampling (simulate along-track)
    if downsample_azimuth and downsample_azimuth > 1:
        f = int(downsample_azimuth)
        tmp = cv2.resize(out, (W//f, H), interpolation=cv2.INTER_AREA)
        out = cv2.resize(tmp, (W, H), interpolation=cv2.INTER_NEAREST)

    return out
