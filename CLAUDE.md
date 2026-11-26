# SSS synthetic-data remake

Paper remake: physics-aware synthetic side-scan sonar (SSS) data for small-object detection.
The spec is [remake_requirements.md](remake_requirements.md). Read it before any design work.
Work on branch `remake`. Code from before the remake (old `detection/`, `classification/`, `synthetic/`, `data/`, notebooks) was deleted from this branch on 2026-11-25; it remains on `main` if its logic is ever needed. `synth/` is the current synthetic-data code.

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
- **Models**: YOLO11s and RT-DETR-l, both COCO-pretrained, Ultralytics. **Update 2026-11-13**: the user decided to switch the main detector to **YOLO11n** (`--model yolon`), matching JMSE 2026, once a 9-run baseline check at 25% confirms that n is not much worse than s.
- **Primary metric: mAP50** (user decision, 2026-11-13). Always report mAP50-95 alongside it.
  - Rationale for the paper: SSS box extents are ambiguous (whether and how much shadow is included) and differ by dataset, and objects are 15–30 px, so mAP50-95 mostly measures agreement with the annotator's box tightness.
  - The switch was made after the go/no-go and did not change its outcome, which still fails under mAP50. Say this openly in the paper.
  - `scripts/summarize.py` defaults to `map50`.
- **Image size**: `imgsz=1024` for every domain, using plain resampling and no super-resolution. Build synthetic images at native resolution.
- **Grayscale**: sonar colour is only a display palette, and AquaScan mixes several palettes. All images, real and synthetic, are single-channel intensity `max(R,G,B)` stored as PNG. Ultralytics replicates the channel to 3, so COCO weights still apply.
- **Synth "ours"** (code in `synth/`):
  - The object bank comes only from that seed's train subset.
  - Each object is an *acoustic signature*: highlight plus its real cast shadow, cut from a crop that extends past the box on the far-range side and stored with the shadow pointing right. The first design (highlight only plus a rendered shadow) failed on Cylinder2 and AquaScan, whose objects are mostly shadow.
  - When pasting, the shadow is mirrored for the port side and stretched by the ratio `shadow_factor(r_target) / shadow_factor(r_source)`.
  - The label box goes through the same transform, so the bbox convention of each domain carries over automatically.
  - Placement uses the seafloor only, avoiding the water column, nadir, flat/no-data regions, and AquaScan's UI mask (`data/aquascan/uimask`).
  - Blending matches the background level by intensity ratio and uses a feathered alpha. Speckle restores the variance lost to resizing, measured on the crop's own seabed pixels.
  - Mine: paste objects only onto backgrounds of the same image-size group.
  - Ablation components: `scale, placement, blending, speckle, shadow`, plus `featherblend`, which switches to Poisson blending. `make_synth.py --ablate X` writes to `synth/ours-noX`.
- **naive**: the raw labelled box at its original size, pasted at a uniform random position with a hard paste.
- **Go/no-go on 2026-11-20**: at 25% real, YOLO, 3 seeds, the mean mAP50-95 of ours must beat naive in at least 2 of 3 domains. If it does not, switch the target to IGARSS.
- **Result (run 2026-11-12/13): FAILED.** Ours beat naive in 1 of 3 domains (Mine +.003, which is noise), and lost on AquaScan (−.030, p=.007). With the user's agreement, the **target is now IGARSS 2027 (deadline 10 Jan 2027)**. The work is now diagnosing why.
  - Ruled out: distorted label boxes. Box aspect ratio and size match real data in every domain.
  - Measured: ours is *closer* to real in-box statistics than naive. Naive has higher contrast and visible box edges, yet it helps more. Working hypothesis: naive acts as a stronger augmentation, and realism alone does not help the detector.
  - User concern: the synthetic shadow may teach YOLO to detect the shadow instead of the object.
  - Diagnostics queued: `ours-noshadow`, `ours-noplacement`, and synth-only training (`train.py --synth-only`) for naive and ours.
  - **Diagnosis result (2026-11-25)**: neither shadow, placement, scale, speckle, nor blending is the cause. Images with ours-pasted objects alone score *below* plain real images (AquaScan mAP50 .502 vs .609), and precision drops (.63 vs .73). The culprit is the core *signature cut by threshold mask*: it yields incomplete blobs without context, which teaches false positives. Naive pastes the whole box and does no harm.
  - **YOLO11n beats YOLO11s at 25% real** in every domain (mAP50: AquaScan .644/.609, Cylinder2 .408/.193, Mine .271/.216). Switched to 11n. All synth comparisons must be redone on 11n.
  - **New direction: forward selection from naive.** `box-<steps>` in `synth/compose.py` pastes the whole box and adds physics steps one at a time: `sea` (seafloor-only placement), `mirror` (shadow away from nadir), `range` (paste near the source ground range), `blend` (feathered box edge plus level ratio). Each step is kept only if it helps.
  - **Forward-selection result (YOLO11n, 25% real, 3 seeds, 2026-11-25)**: no copy-paste variant (naive, box-sea, +mirror, +range, +blend, ours) beats baseline 2 significantly in any domain. Differences of ±.01–.03 mAP50 are below the std (.02–.08). `blend` is the only consistent effect, and it is negative in all 3 domains. Ours is worst on Mine (.196 vs .271). Copy-paste gains seen on 11s vanished on 11n. Next: the generative pilots, and testing at 10% real and/or 5 seeds.
  - **Generative track** (user request, 2026-11-25): box-conditioned inpainting. Diffusion is a LoRA on SD 1.5 inpainting (`scripts/train_inpaint.py`, `scripts/gen_inpaint.py`, `synth/inpaint.py`). The GAN is the official pix2pix networks (`unet_256`, PatchGAN, vanilla loss, L1 100), cloned at `/scratch/.../ext/pix2pix`, in `scripts/train_gan_inpaint.py`. **Plan A (user decision)**: one generator per type, trained on the full train splits of all domains and conditioned on (domain, class); generative results are judged at 100% real, and low-data results count only as an upper bound. Generation verifies each object with the object-vs-seabed judge (`synth/judge.py`, ResNet18, `scripts/train_judge.py`), calibrated at 90% recall on real val objects; the older S/N + YOLO check remains as `--verify detector`. Pass PIL images to diffusers: numpy uint8 inputs are read as [0, 1] floats and give flat grey fills.
  - **The user allows a full redesign of the synth method** (2026-11-13), including dropping the current logic or moving to GAN/diffusion, if it makes the paper work. Before proposing generative models, weigh the data size (only 83–190 objects at 25% real), the 100 GPU-hour budget, and whether the comparison stays fair.

## Data facts (checked 2026-11-12)

- **bbox convention**:
  - Mine: the box covers the object highlight only, and the long cast shadow is outside it.
  - Cylinder2 and AquaScan: the box covers the highlight plus the short attached shadow.
  - Synthetic labels must follow the convention of their domain.
- **Mine 416² near-duplicates**: 416 of the 584 images at 416² are downscaled copies of 1024² images (pHash distance ≤ 6, identical labels). `prepare_data.py` drops these copies (user decision), so Mine has 754 images: 531 train, 113 val, 110 test.
- **Training budget**: 100 virtual epochs × 512 images (3,200 steps), which is the default in `train.py`. In a 200-epoch run, val mAP had mostly plateaued by about epoch 100–150.
- **AquaScan**: one source PNG is truncated (`9bf66ea1-Screenshot_2025-08-03_15.08.49.png`); `prepare_data.py` skips it and logs it in `report.json`.
- **Prepared data**: `/scratch/lt200304-dipmt/paweekorn/data/<domain>/{images,labels,meta.csv}` holds grayscale PNGs built by `scripts/prepare_data.py`. Synthetic sets are in `.../data/<domain>/synth/<method>/s{seed}_p{pct}/`, built by `scripts/make_synth.py` on a CPU node via `scripts/slurm/cpu.sh`. `data_color_old/` and `results/_color_archive/` hold the superseded colour runs.
- **Gap to JMSE 2026** (YOLO11n on Mine, mAP50-95 .648 vs our ~.25): leakage from duplicates does *not* explain it (leaky split .255 vs clean .245). The next hypothesis is that they used positive-only images; this is not tested yet.
- **Workflow tools**: `scripts/summarize.py` prints mean ± std and paired t-tests by seed. `tools/preview_synth.py` renders local naive-vs-ours contact sheets from `local_samples/`, which is gitignored.

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
- Code sync: the LANTA checkout is a git clone of branch `remake`. Uncommitted local work is copied over with `tar cf - <paths> | ssh transfer_lanta 'cd <checkout> && tar xf -'`, because commits wait for the user. The local machine has no rsync.
- Local Python: use `.venv` (gitignored) and set `PYTHONIOENCODING=utf-8`. The console codepage is cp874, and non-ASCII output crashes without it.
- **Cluster etiquette: at most 4 GPUs/jobs running at once** (user request; LANTA is shared). Submit only through `bash scripts/slurm/submit.sh <jobs.txt> [PACK] [sbatch args]`. It adds an array throttle `%4` and `--dependency=singleton`, and it runs PACK (default 3) training lines concurrently on each GPU. Never call `sbatch` on `array.sh` directly.
- **Budget: 1000 LANTA SHr** (raised by the user on 2026-11-25, replacing the earlier 100 GPU-hour cap). Measured rate: 1 GPU-hour ≈ 6 SHr, so the budget is about 160 GPU-hours. `bash scripts/slurm/gpu_hours.sh` prints both units; run it before and after each batch and report it to the user. The SHr figure is project-wide, so other members of `lt200304` count towards it too.
- Each training line writes `results/<run_id>.json`.

## Git

- Never commit or push on your own. Show the diff, wait for the user to approve, and let the user decide when to push.
- Do not put a username, email, or other personal account details in any file that goes into git.

## Code rules

- Real code lives in `.py` files and runs through the CLI and Slurm. Notebooks are for exploration and figures only.
- Use the layout from spec §7: `configs/ data/splits/ synth/ scripts/ tools/ results/`. `tools/` holds exploration and figure scripts (plain `.py`, run locally or over ssh), in place of the spec's `notebooks/`.
- Commit split files. Do not commit images, weights, or generated data.
- Every run must be reproducible from (domain, real fraction, synth method, ratio, model, seed).
