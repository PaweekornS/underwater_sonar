from utils import *
import matplotlib.pyplot as plt

ref_path  = "klsg_dataset/aircraft/000001_jpg.rf.2fa1cc60e74968e8a2d4710607582135.jpg"
mask_path = "optical/aircraft/aircraft_1.jpg"

out = generate_semisynthetic_from_mask(
    mask_path=mask_path,
    reference_path=ref_path,
    out_w=512, out_h=256,
    obj_height_frac=0.6,
    x_offset_frac=0.15,
    shadow_dx=35, shadow_dy=15, shadow_scale=1.0,
    downsample_azimuth=0
)

plt.figure(figsize=(10,4))
plt.imshow(out, cmap="gray")
plt.axis('off')
plt.title("Semisynthetic sidescan sonar (from binary mask)")
plt.show()
