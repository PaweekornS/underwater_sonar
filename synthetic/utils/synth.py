from optical import sample_valid_position, filter_object, rotate_object, construct_shadow
from output import save_output_image, save_yolo_annotation, bbox_from_mask, save_yolo_multi_annotation

import numpy as np
import random
import json
import cv2
import os

class_json_path = os.path.join(os.path.dirname(__file__), "..", "class.json")
with open(class_json_path, "r") as f:
    CLASS = json.load(f)
    f.close()
    
def composite_foreground(bg_img, fg_img, shadow, top, left):
    out = bg_img.copy()

    h, w = fg_img.shape
    roi = out[top:top+h, left:left+w]

    obj_mask = fg_img > 0
    shadow_mask = shadow > 0

    # --- Shadow: anchor to darkest background ---
    shadow_norm = shadow.astype(np.float32) / 255.0

    bg_min = np.percentile(bg_img, 2)  # robust noise floor

    roi_float = roi.astype(np.float32)

    roi_float[shadow_mask] = (
        (1 - shadow_norm[shadow_mask]) * roi_float[shadow_mask] +
        shadow_norm[shadow_mask] * bg_min
    )

    roi = roi_float.astype(np.uint8)

    # --- Object overwrite ---
    roi[obj_mask] = fg_img[obj_mask]

    out[top:top+h, left:left+w] = roi
    return out

def place_single_object(
    bg_path, segment_path, 
    filename, side="random",
    classes=CLASS, 
    output_dir="./yolo_data", 
    save=True
):
    bg_img = cv2.imread(bg_path, 0)
    bg_img = cv2.resize(bg_img, (640, 640), cv2.INTER_NEAREST)
    
    optical_img = filter_object(segment_path)

    # --- rotation ---
    optical_img = rotate_object(optical_img)

    fg_h, fg_w = optical_img.shape
    bg_h, bg_w = bg_img.shape

    top, left = sample_valid_position(
        bg_shape=(bg_h, bg_w),
        fg_shape=(fg_h, fg_w),
        side=side
    )
    
    # --- shadow construction ---    
    bg_center_x = bg_w / 2
    actual_side = "left" if (left + fg_w/2) < bg_center_x else "right"

    shadow = construct_shadow(optical_img, direction=actual_side)
    
    # --- create output ---
    composed = composite_foreground(bg_img, optical_img, shadow, top, left)

    combined_mask = (optical_img > 0) | (shadow > 0)
    x0, y0, x1, y1 = bbox_from_mask(combined_mask)
    bbox_bg = (x0 + left, y0 + top, x1 + left, y1 + top)

    # --- save artifact ---
    class_label = os.path.basename(segment_path).split("-")[0]
    class_id = classes[class_label]

    if save:
        save_output_image(composed, filename, f"{output_dir}/images")
        save_yolo_annotation(bbox_bg, bg_path, filename, f"{output_dir}/labels", class_id=class_id)

    return composed, bbox_bg

def place_multiple_objects(
    bg_paths,
    segment_paths,
    filename,
    max_objects=5,
    side="random",
    classes=CLASS,
    output_dir="./yolo_data",
    allow_overlap=False,
    save=True,
):
    """
    Multi-object semi-synthetic SSS generation.
    
    Returns: (composed, bboxes, class_ids)
    """
    
    bg_path = random.choice(bg_paths)

    # --- base background ---
    bg_img = cv2.imread(bg_path, 0)
    bg_img = cv2.resize(bg_img, (640, 640), cv2.INTER_NEAREST)

    composed = bg_img.copy()
    bg_h, bg_w = composed.shape

    # Track results
    bboxes = []
    class_ids = []

    # Optional occupancy mask (for overlap control)
    occupancy = np.zeros_like(bg_img, dtype=np.uint8)

    num_objects = random.randint(1, max_objects)

    for i in range(num_objects):
        segment_path = random.choice(segment_paths)

        # --- load + extract ---
        optical_img = filter_object(segment_path)
        optical_img = rotate_object(optical_img)

        fg_h, fg_w = optical_img.shape

        # Skip too large objects
        if fg_h >= bg_h or fg_w >= bg_w:
            continue

        # --- sample position ---
        top, left = sample_valid_position(
            bg_shape=(bg_h, bg_w),
            fg_shape=(fg_h, fg_w),
            side=side
        )

        # --- shadow ---
        bg_center_x = bg_w / 2
        actual_side = "left" if (left + fg_w/2) < bg_center_x else "right"

        shadow = construct_shadow(optical_img, direction=actual_side)

        # --- overlap check (optional but important) ---
        if not allow_overlap:
            obj_mask = (optical_img > 0) | (shadow > 0)

            occ_roi = occupancy[top:top+fg_h, left:left+fg_w]

            overlap = np.sum((occ_roi > 0) & obj_mask)

            if overlap > 0:
                continue  # reject placement

        # --- composite ---
        composed = composite_foreground(
            composed,
            optical_img,
            shadow,
            top,
            left
        )

        # --- update occupancy ---
        obj_mask = ((optical_img > 0) | (shadow > 0)).astype(np.uint8)
        occupancy[top:top+fg_h, left:left+fg_w][obj_mask > 0] = 1

        # --- bbox ---
        x0, y0, x1, y1 = bbox_from_mask(obj_mask)
        bbox_bg = (x0 + left, y0 + top, x1 + left, y1 + top)

        bboxes.append(bbox_bg)

        # --- class ---
        class_label = os.path.basename(segment_path).split("-")[0]
        class_id = classes[class_label]
        class_ids.append(class_id)

    # --- save outputs ---
    if save:
        save_output_image(composed, filename, f"{output_dir}/images")
        save_yolo_multi_annotation(
            bboxes,
            class_ids,
            composed.shape,
            filename,
            f"{output_dir}/labels"
        )

    return composed, bboxes, class_ids
