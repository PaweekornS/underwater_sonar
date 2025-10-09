# 🐠 Side-Scan Sonar Image Synthesis for Enhanced Underwater Object Detection

> **Goal:** Prove that semisynthetic and generative sonar images can enhance side-scan sonar object detection performance and improve model generalization on unseen optical-like targets.

---

## 📘 Project Overview

This project explores **two parallel research pipelines**:

1. **Synthetic Image Generation** — producing artificial side-scan sonar images via multiple generative strategies.
2. **Object Detection** — benchmarking YOLO-based detectors trained on real, synthetic, and hybrid datasets.

By combining these efforts, we aim to **quantify the contribution of synthetic data** to underwater object detection performance and demonstrate **cross-domain generalization** (e.g., detecting optical-like objects composited into real sonar backgrounds).

---

## ⚙️ Pipeline Summary

### 1️⃣ Synthetic Image Generation

We employ three distinct generative approaches to create sonar-like imagery:

| Approach | Description | Goal |
|-----------|--------------|------|
| **Procedural (Semisynthetic)** | Classical simulation using reference sonar statistics (Weibull-based intensity distribution) and optical object masks. | Quickly expand training data for low-sample classes (e.g., victims, airplanes). |
| **GAN-based** | Learn sonar texture generation via **GANs** (e.g., StyleGAN3 / Pix2Pix / ADA-StyleGAN). | Capture high-frequency sonar texture realism. |
| **Diffusion-based** | Generate or enhance sonar imagery via **diffusion models** (e.g., DDPM / Cold Diffusion). | Preserve global sonar structure while improving detail and variation. |

Each generated dataset is analyzed in terms of **visual realism**, **statistical similarity** (e.g., FID score), and **utility for object detection**.

---

### 2️⃣ Preprocessing

We will investigate preprocessing techniques that may enhance model robustness and image clarity before training:

- **Noise filtering** (speckle or Gaussian noise suppression)
- **Contrast enhancement** (CLAHE — Contrast Limited Adaptive Histogram Equalization)
- **Super-resolution** (optional; may use ESRGAN or diffusion-based upscalers)

These are experimental steps to assess their impact on detection accuracy.

---

### 3️⃣ Object Detection

We use the **YOLO framework** (Ultralytics YOLOv8 / YOLOv5) as the baseline detector due to its real-time performance and robustness.

#### Experiments

| Experiment | Training Dataset | Objective |
|-------------|------------------|------------|
| **E1** | Real sonar dataset only (KLSG, SDVDs) | Baseline real-data detection performance. |
| **E2** | Synthetic dataset only (from Procedural, GAN, Diffusion) | Evaluate how much synthetic data alone can represent sonar features. |
| **E3** | Mixed dataset (Real + Synthetic) | Test synthetic data as augmentation for improving generalization. |

#### Evaluation Metrics
- mAP (mean Average Precision)
- Precision / Recall
- F1-score
- Inference time
- Cross-domain detection on mixed scenes (real + optical object composites)

---

## 🧩 Dataset Sources

**KLSG:** https://drive.google.com/file/d/1lao8VSbycjlSpctpaeKO0Vfn8H-TGEtj/view?usp=sharing \
**Mine:** https://www.kaggle.com/datasets/sierra022/sonar-imaging-mine-detection \
**Drowning victim:** https://github.com/AJaszcz/SDVDs-Sonar-Drowned-Victim-Datasets/tree/v2.0.1 

All datasets are pre-split into **train / val / test** for reproducibility.

---

## 🧪 Expected Outcomes

- Quantitative proof that **synthetic sonar data** can **enhance YOLO detection accuracy** compared to real-only training.
- Demonstration of **generalization**: the model trained on real + synthetic data should detect **optical objects inserted into real sonar scenes**.
- Comparative report between **procedural, GAN-based, and diffusion-based** synthetic data performance.

 ### Reference paper
 https://ieeexplore.ieee.org/document/9026963
