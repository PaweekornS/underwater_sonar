from utils import *
import matplotlib.pyplot as plt

from tqdm import tqdm
import random
import glob

ROOT_DIR = "C:/Users/User/Downloads/underwater_sonar"

ref_path  = glob.glob(f"{ROOT_DIR}/klsg_dataset/plane/*.png")
mask_images = glob.glob(f"{ROOT_DIR}/masked/plane/*.jpg")

counter = 0
for path in tqdm(mask_images):
    ref = random.choice(ref_path)
    for i in range(10):
        out = semi_synthetic(
            ref_path=ref,
            mask_path=path,
            out_path=f"{ROOT_DIR}/synthetic/result/synth_plane-{str(counter+1).zfill(3)}.png"
        )
        counter += 1
        