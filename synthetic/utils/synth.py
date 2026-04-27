from utils.optical import sample_valid_position, filter_object, rotate_object, construct_shadow
from utils.output import save_output_image, save_yolo_annotation, bbox_from_mask, save_yolo_multi_annotation

import numpy as np
import random
import json
import cv2
import os

class_json_path = os.path.join(os.path.dirname(__file__), "..", "class.json")
with open(class_json_path, "r") as f:
    CLASS = json.load(f)
    f.close()
    
def remove_predefined_objects(image_path):
    """
    Reads a YOLO label file, masks the objects in the image with 0 (black),
    and returns the cleaned background image.
    """
    # 1. Load the background image
    bg_img = cv2.imread(image_path, 0)
    if bg_img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = bg_img.shape

    # 2. Check if label file exists
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    if not os.path.exists(label_path):
        return bg_img  # If no objects, return original image

    # 3. Process labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # YOLO format: class x_center y_center width height
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        _, cx, cy, bw, bh = map(float, parts)

        # 4. Convert normalized to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # 5. Mask out the area (set to 0 for black)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Create a mask for the area
        mask = np.zeros(bg_img.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        # Fill the area with surrounding texture data
        bg_img = cv2.inpaint(bg_img, mask, 3, cv2.INPAINT_TELEA)
    return bg_img
    
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

    # Apply shadow to ROI (as before)
    roi_float[shadow_mask] = (
        (1 - shadow_norm[shadow_mask]) * roi_float[shadow_mask] +
        shadow_norm[shadow_mask] * bg_min
    )

    # --- Soft alpha blending for object edges ---
    mask_float = (obj_mask.astype(np.float32))
    # Use a fairly large blur to soften edges; kernel tuned for 640x640 images
    alpha = cv2.GaussianBlur(mask_float, (31, 31), 0)

    # Normalize alpha to [0,1] safely
    amax = alpha.max()
    if amax > 0:
        alpha = alpha / amax

    # Convert images to float for blending
    fg_float = fg_img.astype(np.float32)

    # Blend: alpha * fg + (1-alpha) * roi
    blended = (alpha * fg_float) + ((1.0 - alpha) * roi_float)

    # Where there is no object at all, keep original ROI to avoid introducing artifacts
    no_obj = (mask_float == 0)
    blended[no_obj] = roi_float[no_obj]

    roi = np.clip(blended, 0, 255).astype(np.uint8)

    out[top:top+h, left:left+w] = roi
    return out

def place_single_object(
    bg_path, segment_path, 
    filename, side="random",
    classes=CLASS, 
    output_dir="./yolo_data", 
    save=True
):
    bg_img = remove_predefined_objects(bg_path)
    bg_img = cv2.resize(bg_img, (640, 640), cv2.INTER_NEAREST)
    
    optical_img = filter_object(segment_path)

    # --- rotation ---
    optical_img = rotate_object(optical_img)

    fg_h, fg_w = optical_img.shape
    bg_h, bg_w = bg_img.shape

    top, left = sample_valid_position(
        bg_shape=(bg_h, bg_w),
        fg_shape=(fg_h, fg_w),
        side=side,
        threshold=0.2
    )
    
    # --- shadow construction ---    
    bg_center_x = bg_w / 2
    actual_side = "left" if (left + fg_w/2) < bg_center_x else "right"

    shadow = construct_shadow(optical_img, direction=actual_side)
    
    # --- create output ---
    composed = composite_foreground(bg_img, optical_img, shadow, top, left)
    class_label = os.path.basename(segment_path).split("-")[0]
    class_id = classes[class_label]
    
    # --- bbox ---
    combined_mask = (optical_img > 0) | (shadow > 0)
    x0, y0, x1, y1 = bbox_from_mask(combined_mask)
    bbox_bg = (class_id, x0 + left, y0 + top, x1 + left, y1 + top)

    # --- save artifact ---
    if save:
        save_output_image(composed, filename, f"{output_dir}/images")
        save_yolo_annotation(bbox_bg, composed.shape, filename, f"{output_dir}/labels")

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
    bg_img = remove_predefined_objects(bg_path)
    bg_img = cv2.resize(bg_img, (640, 640), cv2.INTER_NEAREST)

    composed = bg_img.copy()
    bg_h, bg_w = composed.shape

    # Track results
    bboxes = []
    class_ids = []

    # Optional occupancy mask (for overlap control)
    occupancy = np.zeros_like(bg_img, dtype=np.uint8)

    num_objects = random.randint(2, max_objects)

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
            side=side,
            threshold=0.2
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

        # --- class ---
        class_label = os.path.basename(segment_path).split("-")[0]
        class_id = classes[class_label]
        class_ids.append(class_id)
        
        # --- bbox ---
        x0, y0, x1, y1 = bbox_from_mask(obj_mask)
        bbox_bg = (class_id, x0 + left, y0 + top, x1 + left, y1 + top)

        bboxes.append(bbox_bg)

    # --- save outputs ---
    if save:
        save_output_image(composed, filename, f"{output_dir}/images")
        save_yolo_multi_annotation(
            bboxes,
            composed.shape,
            filename,
            f"{output_dir}/labels"
        )

    return composed, bboxes, class_ids
