"""Box-conditioned GAN inpainting with the official pix2pix networks (Isola et al., CVPR 2017), trained from
scratch: the comparison for the diffusion model. By default one model for all domains (--domain all),
conditioned on (domain, class) through one-hot maps. It uses the same examples as scripts/train_inpaint.py
(same subset, windows, masks and canonical orientation), so the two differ only in the generator.

Networks and hyper-parameters follow the pix2pix defaults: unet_256 generator with batch norm and dropout,
70x70 PatchGAN ("basic"), vanilla GAN loss, lambda_L1 = 100, Adam lr 2e-4 with beta1 0.5. The discriminator
is conditioned on the generator input, as in pix2pix. Adaptation for box inpainting: the generator input is
[masked window, mask, class maps] and only the masked region of its output is used.

The official repository (BSD licence) is cloned at PIX2PIX below; its commit is recorded in meta.json.

Output: /scratch/.../inpaint_gan/<domain>_s{seed}_p{pct}/{generator.pt, meta.json, samples.png}

Example: python scripts/train_gan_inpaint.py --domain aquascan --seed 0 --real 0.25
"""
import argparse
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from train_inpaint import CLASSES, NAMES, SCRATCH, build_examples, sample  # noqa: E402

PIX2PIX = SCRATCH / "ext" / "pix2pix"   # git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
RES = 256


def load_networks():
    """Import pix2pix's models/networks.py directly, without the package __init__ (which pulls in the
    training framework)."""
    spec = importlib.util.spec_from_file_location("pix2pix_networks", PIX2PIX / "models" / "networks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gen_input(img, mask, cls, n_cls):
    """img, mask: (B,1,H,W) in [-1,1] / {0,1}; cls: (B,) -> generator input [masked, mask, class maps]."""
    onehot = F.one_hot(cls, n_cls).float()[:, :, None, None].expand(-1, -1, *img.shape[-2:])
    return torch.cat([img * (1 - mask), mask, onehot * mask], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["all", *NAMES])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=0, help="accepted for scripts/slurm/array.sh; unused")
    args = ap.parse_args()

    networks = load_networks()
    commit = subprocess.run(["git", "-C", str(PIX2PIX), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pct = round(args.real * 100)
    examples = []
    for d in (NAMES if args.domain == "all" else [args.domain]):
        split = REPO / "data" / "splits" / d / ("train.txt" if pct == 100 else f"train_s{args.seed}_p{pct}.txt")
        examples += build_examples(d, split.read_text().split())
    out = SCRATCH / "inpaint_gan" / f"{args.domain}_s{args.seed}_p{pct}"
    out.mkdir(parents=True, exist_ok=True)
    n_cls = len(CLASSES)  # conditioning over (domain, class) pairs
    in_nc = 2 + n_cls
    print(f"{len(examples)} objects -> {out}  (pix2pix {commit[:7]})", flush=True)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "cuda"
    G = networks.define_G(in_nc, 1, 64, "unet_256", norm="batch", use_dropout=True).to(dev)
    D = networks.define_D(in_nc + 1, 64, "basic", norm="batch").to(dev)
    gan_loss = networks.GANLoss("vanilla").to(dev)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    def batch():
        bs = [sample(examples[rng.randrange(len(examples))], rng) for _ in range(args.batch)]
        im = np.stack([cv2.resize(b[0], (RES, RES), interpolation=cv2.INTER_AREA) for b in bs])
        m = np.stack([cv2.resize(b[1], (RES, RES), interpolation=cv2.INTER_NEAREST) for b in bs])
        return (torch.tensor(im)[:, None].to(dev), torch.tensor(m)[:, None].to(dev),
                torch.tensor([b[2] for b in bs], device=dev))

    for step in range(1, args.steps + 1):
        real, m, cls = batch()
        cond = gen_input(real, m, cls, n_cls)
        fake = real * (1 - m) + G(cond) * m   # only the masked region is generated

        opt_d.zero_grad(set_to_none=True)
        d_loss = 0.5 * (gan_loss(D(torch.cat([cond, real], 1)), True) +
                        gan_loss(D(torch.cat([cond, fake.detach()], 1)), False))
        d_loss.backward()
        opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        adv = gan_loss(D(torch.cat([cond, fake], 1)), True)
        l1 = F.l1_loss(fake, real)
        (adv + args.lambda_l1 * l1).backward()
        opt_g.step()
        if step % 1000 == 0 or step == 1:
            print(f"step {step} d {d_loss.item():.3f} g_adv {adv.item():.3f} l1 {l1.item():.4f}", flush=True)

    torch.save(G.state_dict(), out / "generator.pt")
    with torch.no_grad():  # visual check: rows of [masked input | generated | real]
        real, m, cls = batch()
        fake = real * (1 - m) + G(gen_input(real, m, cls, n_cls)) * m
        rows = [torch.cat([real[i] * (1 - m[i]), fake[i], real[i]], 2) for i in range(min(8, len(real)))]
        grid = ((torch.cat(rows, 1)[0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "samples.png"), grid)
    (out / "meta.json").write_text(json.dumps({**vars(args), "n_objects": len(examples), "res": RES,
                                               "pix2pix_commit": commit, "netG": "unet_256", "netD": "basic",
                                               "gan_mode": "vanilla", "norm": "batch"}, indent=2))


if __name__ == "__main__":
    main()
