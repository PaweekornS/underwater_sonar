import shutil
import os
import cv2

def save_output_image(img, filename, output_dir):
    """Save foreground image to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename + ".png")
    cv2.imwrite(save_path, img)
    return save_path

def save_yolo_annotation(bbox, bg_path, filename, output_dir, class_id=0):
    """Converts a bounding box to YOLO format and saves it to a .txt file."""
    os.makedirs(output_dir, exist_ok=True)

    bg_img = cv2.imread(bg_path, 0)
    bg_shape = bg_img.shape
    img_h, img_w = bg_shape
    x_min, y_min, x_max, y_max = bbox

    # 1. Calculate Center and Size (Absolute)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    box_w = x_max - x_min
    box_h = y_max - y_min

    # 2. Normalize (0 to 1)
    norm_center_x = center_x / img_w
    norm_center_y = center_y / img_h
    norm_width = box_w / img_w
    norm_height = box_h / img_h

    # 3. Format String
    annotation_line = f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"

    # 4. Save to File
    save_path = os.path.join(output_dir, filename + ".txt")
    bg_labels = bg_path.replace("images", "labels").replace(".jpg", ".txt")

    try:
        shutil.copy(bg_labels, save_path)
        with open(save_path, "a") as f:
            f.write(annotation_line)
    except:
        with open(save_path, "w") as f:
            f.write(annotation_line)