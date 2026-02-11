import glob
import cv2
from tqdm import tqdm
import random

from utils.semisynth import refine_mask, place_object_on_sss_background

OBJECT = "victim"
BASE_DIR = "./data/segment_prep"
MASK_FILES = glob.glob(f"{BASE_DIR}/masked/{OBJECT}/*.png")
BACKGROUND_FILES = glob.glob("./data/klsg_dataset/seafloor/*.jpg")
print(len(BACKGROUND_FILES))

for i in tqdm(range(10)):
    bg_path = random.choice(BACKGROUND_FILES)  # background
    mask_path = random.choice(MASK_FILES)
    mask = cv2.imread(mask_path, 0)

    optical_path = mask_path.replace("masked", "resized")
    optical_img = cv2.imread(optical_path, 0)

    canvas = refine_mask(optical_img, mask)
    composed, bbox = place_object_on_sss_background(
        bg_path, optical_path, canvas,
        filename=f"{OBJECT}-{i+1:03d}", side="random",
        output_dir="./synthetic/Procedural/results"
    )
    