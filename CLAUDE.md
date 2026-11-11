# SSS synthetic-data remake

Paper remake: physics-aware synthetic side-scan sonar (SSS) data for small-object detection.
The spec is [remake_requirements.md](remake_requirements.md). Read it before any design work.
Work on branch `remake`. Code from before the remake (`detection/`, `classification/`, `synthetic/`, `data/`) is legacy: reuse its logic, do not extend it.

Reply to the user in Thai. Keep code, comments, commit messages, and the paper in English.

## Decisions already made (do not re-ask)

Settled with the user on 2026-11-12. These override the spec where they differ.

- **Domains** (all full-swath, reported separately, never merged):
  - Cylinder2: 2 classes `cylinder`, `manta`. Raw ids 0 (`cylider`, val/test only) and 1 (`cylinder`, train only) are the same object, so merge them. Raw id 2 is `manta`. Use the official split. Remove near-duplicates across splits from train.
  - Mine: classes `MILCO`, `NOMBO`. Random fixed-seed split, stratified by class and empty images, pHash dedup before splitting. Keep all empty images.
  - AquaScan-1K: class `human`. It is full-swath, but the images are fishfinder screenshots with UI overlays (depth, GPS, menu, scale bars). Keep the UI in the images, because cropping would lose 9–20% of the boxes. Detect a UI mask for each image, and never let synth placement overlap it. There are no empty images.
- **Training configs**:
  1. real only, all augmentation off
  2. real + Ultralytics default augmentation
  3. config 2 + naive copy-paste
  4. config 2 + ours
- **Low-data subsets**: nested (10% ⊂ 25% ⊂ 50% ⊂ 100%). The subset changes with the seed. Every config uses the same subset for the same seed, so results are paired.
- **Models**: YOLO11s and RT-DETR-l, both COCO-pretrained, Ultralytics.
- **Image size**: `imgsz=1024` for every domain, using plain resampling and no super-resolution. Build synthetic images at native resolution.
- **Synth**:
  - The object bank comes only from that seed's train subset.
  - Cut out the object highlight with a mask and drop the real shadow. Render a new shadow that points away from nadir and grows longer with range.
  - Mine: paste objects only onto backgrounds of the same image-size group (416² with 416², 1024² with 1024²).
- **Go/no-go on 2026-11-20**: at 25% real, YOLO, 3 seeds, the mean mAP50-95 of ours must beat naive in at least 2 of 3 domains. If it does not, switch the target to IGARSS.

## Data facts (checked 2026-11-12)

- **bbox convention**:
  - Mine: the box covers the object highlight only, and the long cast shadow is outside it.
  - Cylinder2 and AquaScan: the box covers the highlight plus the short attached shadow.
  - Synthetic labels must follow the convention of their domain.
- **Mine 416² near-duplicates**: 416 of the 584 images at 416² are downscaled copies of 1024² images (pHash distance ≤ 6, identical labels). `prepare_data.py` drops these copies (user decision), so Mine has 754 images: 531 train, 113 val, 110 test.
- **Training budget**: 100 virtual epochs × 512 images (3,200 steps), which is the default in `train.py`. In a 200-epoch run, val mAP had mostly plateaued by about epoch 100–150.
- **AquaScan**: one source PNG is truncated (`9bf66ea1-Screenshot_2025-08-03_15.08.49.png`); `prepare_data.py` skips it and logs it in `report.json`.
- **Prepared data**: `/scratch/lt200304-dipmt/paweekorn/data/<domain>/{images,labels,meta.csv}`, built by `scripts/prepare_data.py`. The images are symlinks to the raw data.

## LANTA (NSTDA supercomputer)

- Access: `ssh transfer_lanta`. Run every remote command as `ssh transfer_lanta '<cmd>'`. Slurm (`sbatch`, `squeue`, `sacct`) works from the transfer node.
- Slurm account: `-A lt200304`. Check the remaining budget with `sbalance`.
- GPU partitions:
  - `gpu`: up to 5 days per job, 4× A100 per node.
  - `gpu-devel`: up to 2 h, for smoke tests.
  - `gpu-limited`: up to 1 day.
- Header the user already uses: `-p gpu -N 1 -c 16 --gpus-per-task=1 --ntasks-per-node=1 -t HH:MM:SS -A lt200304`. Request 16 CPUs per GPU.
- Python environment: `ml Mamba` then `conda activate <env>`. The project env is `unsloth_env` (the user's existing env, with torch 2.10 + cu128 and ultralytics 8.4). Install any missing package into it with `pip install -c <constraints>`, where the constraints file pins the current torch and torchvision versions. Do not create a new env.
- Compute nodes are treated as offline. On the transfer node, download pretrained weights and pip packages ahead of time.
- Paths:
  - Raw data, read-only: `/scratch/lt200304-dipmt/sonar-data/{cylider2,mine-detection,aquaScan}`
  - Code checkout: `/project/lt200304-dipmt/paweekorn/sss-project`
  - Large outputs (synthetic images, runs): `/scratch/lt200304-dipmt/paweekorn/` (do not put them in `/project` or git)
  - Legacy: `/project/lt200304-dipmt/paweekorn/sss_project` is the old project. Leave it alone.
- Code sync: commit, push to GitHub (`origin`), then `git pull` on LANTA. LANTA can reach github.com. The local machine has no rsync.
- Run many seeds or configs as a Slurm job array. Each array task writes `results/<run_id>.json`.

## Git

- Never commit or push on your own. Show the diff, wait for the user to approve, and let the user decide when to push.
- Do not put a username, email, or other personal account details in any file that goes into git.

## Code rules

- Real code lives in `.py` files and runs through the CLI and Slurm. Notebooks are for exploration and figures only.
- Use the layout from spec §7: `configs/ data/splits/ synth/ scripts/ notebooks/ results/`.
- Commit split files. Do not commit images, weights, or generated data.
- Every run must be reproducible from (domain, real fraction, synth method, ratio, model, seed).
