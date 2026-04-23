import numpy as np
import shutil
import os
import cv2

def bbox_from_mask(mask):
    """Compute tight bounding box from binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask, cannot compute bbox.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return x_min, y_min, x_max, y_max

def save_output_image(img, filename, output_dir):
    """Save foreground image to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename + ".png")
    cv2.imwrite(save_path, img)
    return save_path

def save_yolo_annotation(bbox, img_shape, filename, output_dir, class_id=0):
    os.makedirs(output_dir, exist_ok=True)

    img_h, img_w = img_shape
    x_min, y_min, x_max, y_max = bbox

    # --- absolute ---
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    box_w = x_max - x_min
    box_h = y_max - y_min

    # --- normalize ---
    norm_center_x = center_x / img_w
    norm_center_y = center_y / img_h
    norm_width = box_w / img_w
    norm_height = box_h / img_h

    # --- safety clipping ---
    norm_center_x = np.clip(norm_center_x, 0.0, 1.0)
    norm_center_y = np.clip(norm_center_y, 0.0, 1.0)
    norm_width    = np.clip(norm_width,    0.0, 1.0)
    norm_height   = np.clip(norm_height,   0.0, 1.0)

    annotation_line = f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"

    save_path = os.path.join(output_dir, filename + ".txt")

    # overwrite (safe default)
    with open(save_path, "w") as f:
        f.write(annotation_line)
            
def save_yolo_multi_annotation(bboxes, class_ids, img_shape, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename + ".txt")

    img_h, img_w = img_shape

    with open(path, "w") as f:
        for bbox, class_id in zip(bboxes, class_ids):
            x_min, y_min, x_max, y_max = bbox

            cx = (x_min + x_max) / 2 / img_w
            cy = (y_min + y_max) / 2 / img_h
            w  = (x_max - x_min) / img_w
            h  = (y_max - y_min) / img_h

            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
