# Synthetic Side-Scan Sonar for Small-Object Detection — Project Requirements

Remake of senior project into a conference paper. This file is the agreed spec; use it as context for implementation.

## 1. Why the remake

- Original results mixed **KLSG** (cropped, large, clear objects) with **Cylinder2** (full-swath, very small objects).
- The ~80% mAP was carried by KLSG; full-swath Cylinder2 only reached 60–70%.
- Other weaknesses: inconsistent test sets (cropped vs full), single runs with no variance, no augmentation baseline, contradictory per-class results.

## 2. Goal

- **Venue:** OCEANS 2027 Aberdeen — paper deadline **7 Dec 2026** (tentative; check official CFP, abstract may come first).
- **Fallback:** IGARSS 2027 — deadline **10 Jan 2027**, same protocol.
- **Core claim:** Physics-aware synthetic data improves object detection on **full-swath** SSS more than standard augmentation and naive copy-paste, especially in **low-data** regimes.
- Out of scope (future work): generative (diffusion) object patches, layout-conditioned full-swath generation.

## 3. Datasets

| Domain | Classes | Split |
|---|---|---|
| Cylinder2 | cylinder | official train/val/test |
| Mine dataset (~1170 images) | MILCO, NOMBO | fixed-seed split |
| AquaScan-1K (Zenodo 18771165, MIT) | human | fixed-seed split |

Rules:
- Full-swath images only. **Verify AquaScan-1K is full-swath**; drop it if it is cropped.
- Classes follow the target domain only. **KLSG is not used** (optional out-of-domain ablation only if time allows).
- Report results **per domain**; never merge into a single number.
- Without an official split: fixed seed, same split for every config, and remove near-duplicates across splits with perceptual hashing before splitting. Prefer split by track/survey if metadata exists.
- Commit split files to the repo.
- Check each dataset's bbox convention (does the box include the shadow?) and follow it per domain.

## 4. Synthetic pipeline

Approach: physics-aware cut & paste, in-domain.

- **Object bank:** crops from the **train split of the same domain** using existing bboxes. In low-data runs, use only objects from that subset (e.g. 10% run → objects from the 10% only). No test/val leakage.
- **Backgrounds:** empty train images (if any) + train images that already contain objects (keep original labels; new boxes must not overlap existing ones).
- **Objects per image & size:** sampled from the train distribution of that domain (scale matching).
- **Placement:** seafloor region only, never water column / nadir.
- **Shadow:** direction points away from nadir (port/starboard aware); **length grows with range**.
- **Blending:** feathered edges + match local background mean/std (main). Poisson blending as ablation.
- **Noise:** multiplicative speckle (Rayleigh/Gamma), parameters estimated from each domain's real background.
- **Labels:** bbox convention per domain (see §3).
- Existing code already has placement-on-seafloor and shadow direction, but is **hard-coded** → refactor into configurable modules.

## 5. Experiments

- **Detectors:** YOLO (Ultralytics) + RT-DETR (Ultralytics).
- **Baselines** (synthetic must beat all):
  1. Real only
  2. Real + standard augmentation (flip, brightness, mosaic)
  3. Real + naive copy-paste (no shadow, noise, scale matching, blending)
  4. Real + ours
- **Low-data:** real 10 / 25 / 50 / 100%.
- **Synthetic ratio:** 1:1 to real as default; sweep 0.5 / 1 / 2 / 4 at 25% real.
- **Fairness:** equal number of training **iterations** across configs; checkpoint selected on **real val only**.
- **Seeds:** 5 for main results, 3 for ablations.
- **Ablation (leave-one-out):** scale, placement, blending, speckle, shadow — at 25% real, YOLO only, all 3 domains.
- **Compute:** LANTA, up to 4× A100 per job, Slurm.

## 6. Evaluation

- **Detection:** mAP50-95 (primary), mAP50, recall. Mean ± std over seeds, paired t-test vs baselines.
- **Image quality:**
  - KID and CMMD, on object-centered patches (primary) and whole images.
  - Real-vs-synthetic classifier (small CNN): accuracy near 50% = more realistic.
  - Do **not** use SSIM/PSNR (no paired ground truth). Note CLIP domain mismatch for CMMD in limitations.
- **Error analysis:** results by object size (small/medium), by range (near/far from nadir), qualitative examples of wins and failures.

## 7. Code structure

Real code in `.py` (run via CLI/Slurm); notebooks only for exploration and figures. Git from day one.

```
project/
├── configs/        # yaml per experiment
├── data/splits/    # fixed split files
├── synth/          # placement, shadow, speckle, blend
├── scripts/        # train.py, eval.py, slurm jobs
├── notebooks/      # exploration & plots
└── results/        # per-run csv/json
```

Example: `python scripts/train.py --domain mine --real 0.25 --synth ours --ratio 1 --model yolo --seed 3`

## 8. Reproducibility

Publish code, splits, configs, and seeds on GitHub at submission; link in paper.

## 9. Timeline (deadline 7 Dec 2026)

| Week | Dates | Work |
|---|---|---|
| 1 | 26 Oct – 1 Nov | Inspect datasets (AquaScan full-swath? bbox conventions), dedup, splits. Co-author: related work, CFP details |
| 2 | 2 – 8 Nov | Refactor synthetic pipeline (§4) |
| 3 | 9 – 15 Nov | Main experiments. **Checkpoint:** if ours doesn't beat naive copy-paste → switch to IGARSS |
| 4 | 16 – 22 Nov | Remaining experiments + ablations |
| 5 | 23 – 29 Nov | Image-quality metrics, error analysis, writing |
| 6 | 30 Nov – 7 Dec | Revise and submit |

## 10. Open items

- [ ] Confirm OCEANS 2027 CFP dates and format (abstract vs full paper, page limit).
- [ ] Verify AquaScan-1K is full-swath.
- [ ] Check bbox conventions of all 3 datasets.
- [ ] Check whether datasets contain empty (object-free) images.
- [ ] Check for track/survey metadata for splitting.
