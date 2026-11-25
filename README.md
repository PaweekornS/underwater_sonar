# Physics-guided synthetic data for side-scan sonar object detection

Does physics-guided synthetic data help detect small objects in **full-swath** side-scan sonar (SSS),
beyond standard augmentation and naive copy-paste, especially when real data is scarce?

## Datasets

All three are full-swath SSS. Each domain is trained and reported separately.

| Domain | Classes | Images | Split |
| --- | --- | --- | --- |
| Cylinder2 (Roboflow, CC BY 4.0) | cylinder, manta | 478 | official train/val/test, with near-duplicates removed from train |
| Mine detection ([figshare 24574879](https://figshare.com/articles/dataset/_i_Side-scan_sonar_imaging_for_Mine_detection_i_/24574879)) | MILCO, NOMBO | 754 after removing downscaled copies | fixed-seed stratified, pHash dedup |
| AquaScan-1K (Zenodo 18771165, MIT) | human | 1032 | fixed-seed stratified, pHash dedup |

Images are converted to single-channel intensity, because sonar colour is only a display palette.
The split files and low-data subsets are committed in [data/splits/](data/splits/): for each of 5 seeds,
the subsets are nested (10% ⊂ 25% ⊂ 50%).

## Pipeline

```text
scripts/prepare_data.py   raw datasets -> grayscale PNG + unified YOLO labels + split files
scripts/make_synth.py     synthetic images for one (domain, seed, real %, method) cell
scripts/train.py          one detector run (equal optimizer steps for every config) -> results/<run_id>.json
scripts/summarize.py      mean ± std per config, paired t-tests by seed
```

Synthetic methods ([synth/](synth/)):

- `naive`: paste the labelled box as-is at a random position.
- `box-<steps>`: naive plus physics steps.
  - `sea`: seafloor-only placement, avoiding the water column, nadir, no-data regions and UI overlays.
  - `mirror`: the shadow points away from nadir.
  - `range`: the box is pasted near its source ground range.
  - `blend`: feathered edge and level matching.
- `ours`: acoustic-signature cut with a range-scaled shadow. This was the first design; it is kept as a
  negative result.

Detectors: YOLO11n (main) and RT-DETR-l, both from Ultralytics and COCO-pretrained, at `imgsz=1024`.
The primary metric is mAP50, and mAP50-95 is reported alongside it.

## Running on Slurm (LANTA)

```bash
# CPU: data prep or synthetic generation, one command per line
sbatch --dependency=singleton scripts/slurm/cpu.sh commands.txt
# GPU: training lines, 3 per GPU, at most 4 GPUs at once
bash scripts/slurm/submit.sh jobs.txt 3
bash scripts/slurm/gpu_hours.sh   # GPU-hours used so far
```

Example training line:

```bash
python scripts/train.py --domain mine --real 0.25 --synth box-sea-mirror --ratio 1 --aug on --model yolon --seed 0
```

## Local preview

```bash
python notebooks/preview_synth.py local_samples/previews   # naive vs physics contact sheets
```

`local_samples/` holds a few train images per domain and is not committed.
