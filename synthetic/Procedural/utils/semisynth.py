import numpy as np
import cv2
import os

print(os.getcwd())

from utils.output import save_output_image, save_yolo_annotation
from utils.position import rotate_object_and_mask, sample_valid_position, bbox_from_mask
from utils.blending import composite_foreground, match_local_stats

CLASS_MAP = {
    "plane": 0,
    "fish": 1,
    "other": 2,
    "ship": 3,
    "victim": 4,
    "cylider": 5,
    "manta": 6
}

# ==============
# main utility
# ==============

def refine_mask(optical_img, mask):
    # Apply the conditional conversion to the mask
    mask[(mask > 0) & (mask < 50)] = 0
    mask[(mask > 200) & (mask < 255)] = 255
    mask[(mask >= 50) & (mask <= 200)] = 127

    # Visualize the modified mask
    canvas = np.zeros_like(optical_img)
    canvas[mask == 255] = optical_img[mask == 255]

    # check masking area
    ys, xs = np.where(mask == 255)
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    if width*height > mask.size / 3:
        canvas = cv2.resize(canvas, (128, 128))

    return canvas

def place_object_on_sss_background(bg_path, optical_path, fg_mask, filename, side="random", output_dir="/content/yolo_data/train"):
    """Place object+shadow on SSS background"""
    bg_img = cv2.imread(bg_path, 0)
    bg_img = cv2.resize(bg_img, (640, 640), cv2.INTER_NEAREST)
    
    fg_img = cv2.imread(optical_path, 0)
    fg_img, fg_mask = rotate_object_and_mask(fg_img, fg_mask)

    fg_h, fg_w = fg_img.shape
    bg_h, bg_w = bg_img.shape

    # Sample valid position
    top, left = sample_valid_position(
        bg_shape=(bg_h, bg_w),
        fg_shape=(fg_h, fg_w),
        side=side
    )

    # Intensity Matching
    fg_img_matched = match_local_stats(bg_img, fg_img, fg_mask, top, left)
    blurred_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0) # Edge  blending
    composed = composite_foreground(bg_img, blurred_mask, fg_img_matched, top, left)

    # Bounding box in foreground coords
    x0, y0, x1, y1 = bbox_from_mask(fg_mask)
    bbox_bg = (
        x0 + left,
        y0 + top,
        x1 + left,
        y1 + top
    )

    # save output
    class_label = os.path.basename(optical_path).split("-")[0]
    save_output_image(composed, filename, f"{output_dir}/images")
    save_yolo_annotation(bbox_bg, bg_path, filename, f"{output_dir}/labels", class_id=CLASS_MAP[class_label])

    return composed, bbox_bg
