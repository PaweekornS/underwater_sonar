import matplotlib.pyplot as plt
import cv2

def visualize_bbox(image, bbox, color=(255, 0, 0), thickness=2, show=True, save_path=None):
    """
    Draws a bounding box on the image.
    """
    x0, y0, x1, y1 = bbox

    # If grayscale, convert to BGR for colored drawing
    if len(image.shape) == 2:
        image_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_vis = image.copy()

    # Draw rectangle
    cv2.rectangle(image_vis, (x0, y0), (x1, y1), color, thickness)

    # Optional display
    if show:
        plt.imshow(image_vis)
        plt.show()

def visualize_bboxes(
    image,
    bboxes,
    color=(255, 0, 0),
    thickness=2,
    show=True,
    save_path=None,
    class_names=None
):
    """
    Draw multiple bounding boxes on an image.

    Args:
        image: Input image (H, W) or (H, W, 3)
        bboxes: List of bounding boxes
                Each bbox can be:
                - (x0, y0, x1, y1)
                - (x0, y0, x1, y1, class_id)
                - (x0, y0, x1, y1, class_id, score)
        color: Default box color
        thickness: Line thickness
        show: Whether to display the image
        save_path: Optional path to save the result
        class_names: Optional list mapping class_id → class name
    """

    # Handle grayscale
    if len(image.shape) == 2:
        image_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_vis = image.copy()

    for bbox in bboxes:
        # Unpack flexibly
        if len(bbox) == 4:
            x0, y0, x1, y1 = bbox
            label = None
        elif len(bbox) == 5:
            x0, y0, x1, y1, class_id = bbox
            label = str(class_id)
            if class_names:
                label = class_names[class_id]
        elif len(bbox) == 6:
            x0, y0, x1, y1, class_id, score = bbox
            label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        else:
            raise ValueError("Invalid bbox format")

        # Draw rectangle
        cv2.rectangle(image_vis, (x0, y0), (x1, y1), color, thickness)

        # Draw label if exists
        if label:
            cv2.putText(
                image_vis,
                label,
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

    # Save if needed
    if save_path:
        cv2.imwrite(save_path, image_vis)

    # Show if needed
    if show:
        plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        