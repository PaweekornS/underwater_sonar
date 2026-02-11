import numpy as np
import glob
import cv2
import os

from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch


# Constraints
REAL_DIR = "./data/..."
FAKE_DIR = "./synthetic/Procedural/results/..."

# ===============
# Utility
# ===============
def calculate_metrics(real_dir, fake_dir, device, batch_size=32):
    try:
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=batch_size,
            device=device,
            dims=2048
        )
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid = None

    # Calculate SSIM and PSNR
    real_images_list = glob.glob(os.path.join(real_dir, "*.png")) + glob.glob(os.path.join(real_dir, "*.jpg"))
    # Fix: Change *.jpg to *.png for fake_images_list
    fake_images_list = glob.glob(os.path.join(fake_dir, "*.png")) + glob.glob(os.path.join(fake_dir, "*.jpg"))

    ssim_scores = []
    psnr_scores = []

    for real_img_path, fake_img_path in zip(real_images_list, fake_images_list):
        real_img_np = cv2.imread(real_img_path, 0) # SSIM/PSNR often calculated on grayscale
        fake_img_np = cv2.imread(fake_img_path, 0)

        # Ensure images have the same dimensions before calculating SSIM/PSNR
        if real_img_np.shape == fake_img_np.shape:
            ssim_scores.append(ssim(real_img_np, fake_img_np, data_range=255))
            psnr_scores.append(psnr(real_img_np, fake_img_np, data_range=255))
        else:
            print(f"Skipping SSIM/PSNR for {os.path.basename(real_img_path)} and {os.path.basename(fake_img_path)} due to dimension mismatch.")


    avg_ssim = np.mean(ssim_scores) if ssim_scores else -1
    avg_psnr = np.mean(psnr_scores) if psnr_scores else -1

    return fid, avg_ssim, avg_psnr

# ===============
# Main pipeline
# ===============
if __name__ == "__main__":
    fid, ssim_score, psnr_score = calculate_metrics(
        REAL_DIR, FAKE_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32
    )

    print(f"FID: {fid:.4f}")
    print(f"SSIM: {ssim_score:.4f}")
    print(f"PSNR: {psnr_score:.4f}")