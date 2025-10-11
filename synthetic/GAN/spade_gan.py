from skimage.filters import threshold_multiotsu
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------
# Segmentation
# ---------------------------

def apply_multiotsu(image_path, num_classes=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    thresholds = threshold_multiotsu(img, classes=num_classes)

    # Digitize the image based on the thresholds
    regions = np.digitize(img, bins=thresholds)

    segmented_img = (regions * (255 // (num_classes - 1))).astype(np.uint8)
    return segmented_img / 127.

# ---------------------------
# Utils
# ---------------------------

def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if getattr(m, "weight", None) is not None and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def one_hot(seg: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer HxW segmentation (0..K-1) to one-hot [B, K, H, W].
    If seg already one-hot, returns as-is.
    """
    if seg.dim() == 4 and seg.size(1) == num_classes:
        return seg
    # seg: [B,1,H,W] or [B,H,W]
    if seg.dim() == 3:
        seg = seg.unsqueeze(1)
    b, _, h, w = seg.shape
    out = torch.zeros(b, num_classes, h, w, device=seg.device, dtype=torch.float32)
    return out.scatter_(1, seg.long(), 1.0)


# ---------------------------
# SPADE Block
# ---------------------------

class SPADE(nn.Module):
    """
    SPADE: Spatially-Adaptive (DE)normalization.
    We modulate an InstanceNorm2d with gamma, beta predicted from segmentation map.
    """
    def __init__(self, norm_nc: int, label_nc: int, ks: int = 3, hidden: int = 128):
        super().__init__()
        pw = ks // 2
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, eps=1e-5)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta  = nn.Conv2d(hidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # seg: [B, label_nc, H, W], resize to x spatial dims if needed
        if seg.shape[-2:] != x.shape[-2:]:
            seg = F.interpolate(seg, size=x.shape[-2:], mode="nearest")
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    """
    Simple residual block with SPADE on both convs (no spectral norm here for simplicity).
    """
    def __init__(self, fin: int, fout: int, label_nc: int):
        super().__init__()
        fmiddle = min(fin, fout)

        self.learned_shortcut = (fin != fout)
        self.spade_1 = SPADE(fin, label_nc)
        self.conv_1  = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)

        self.spade_2 = SPADE(fmiddle, label_nc)
        self.conv_2  = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.spade_s = SPADE(fin, label_nc)
            self.conv_s  = nn.Conv2d(fin, fout, kernel_size=1, padding=0)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.spade_s(x, seg)
            x_s = self.conv_s(self.relu(x_s))
            return x_s
        return x

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x_s = self.shortcut(x, seg)

        dx = self.spade_1(x, seg)
        dx = self.conv_1(self.relu(dx))

        dx = self.spade_2(dx, seg)
        dx = self.conv_2(self.relu(dx))

        return x_s + dx


# ---------------------------
# Pixel Distribution Predictor (μ, logσ)
# ---------------------------

class PixelDistributionPredictor(nn.Module):
    """
    CNN that extracts pixel/style stats and predicts mean, logvar for latent z.
    """
    def __init__(self, in_nc: int = 1, z_dim: int = 256):  # SSS often grayscale -> in_nc=1; change if RGB=3
        super().__init__()
        nf = 64
        self.features = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf*2, 3, 2, 1), nn.BatchNorm2d(nf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2, nf*4, 3, 2, 1), nn.BatchNorm2d(nf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4, nf*8, 3, 2, 1), nn.BatchNorm2d(nf*8), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.mu     = nn.Linear(nf*8, z_dim)
        self.logvar = nn.Linear(nf*8, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        f = self.features(x).view(b, -1)
        mu = self.mu(f)
        logvar = self.logvar(f)
        return mu, logvar


# ---------------------------
# Generator (Pix2pixHD-like upsampling + SPADE)
# ---------------------------

class SPADEGenerator(nn.Module):
    """
    Start from latent z, project to small spatial map, then upsample with SPADEResBlocks to 256x256.
    """
    def __init__(self, z_dim: int, out_nc: int, label_nc: int, ngf: int = 64, out_size: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.label_nc = label_nc
        self.out_size = out_size

        self.fc = nn.Linear(z_dim, 16*ngf*4*4)  # start at 4x4 with 16*ngf channels

        ch = 16 * ngf
        self.block_4  = SPADEResBlock(ch, 8*ngf,  label_nc)
        self.block_8  = SPADEResBlock(8*ngf,  8*ngf,  label_nc)
        self.block_16 = SPADEResBlock(8*ngf,  4*ngf,  label_nc)
        self.block_32 = SPADEResBlock(4*ngf,  2*ngf,  label_nc)
        self.block_64 = SPADEResBlock(2*ngf,  1*ngf,  label_nc)
        self.block_128= SPADEResBlock(1*ngf,  1*ngf//2, label_nc)
        self.block_256= SPADEResBlock(1*ngf//2, 1*ngf//4, label_nc)

        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf//4, out_nc, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        z: [B, z_dim]
        seg: [B, label_nc, H, W] (will be down/up-sampled internally)
        output: [B, out_nc, 256, 256] by default
        """
        b = z.size(0)
        x = self.fc(z).view(b, -1, 4, 4)

        # Upsample progressively to reach out_size
        def up(x): return F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.block_4(x, seg)     # 4x4 -> 4x4
        x = up(x)
        x = self.block_8(x, seg)     # 8x8
        x = up(x)
        x = self.block_16(x, seg)    # 16x16
        x = up(x)
        x = self.block_32(x, seg)    # 32x32
        x = up(x)
        x = self.block_64(x, seg)    # 64x64
        x = up(x)
        x = self.block_128(x, seg)   # 128x128
        x = up(x)
        x = self.block_256(x, seg)   # 256x256

        out = self.to_rgb(x)
        return out


# ---------------------------
# PatchGAN Discriminator + Multi-scale
# ---------------------------

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN: takes concatenated [image, seg_onehot] as input.
    Returns logits and intermediate features for feature matching.
    """
    def __init__(self, in_nc: int, ndf: int = 64, n_layers: int = 4):
        super().__init__()
        kw = 4
        pw = 1

        sequence = []
        # first layer (no norm)
        sequence += [
            nn.Conv2d(in_nc, ndf, kernel_size=kw, stride=2, padding=pw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf*nf_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=pw),
                nn.InstanceNorm2d(ndf*nf_mult, affine=False, eps=1e-5),
                nn.LeakyReLU(0.2, True),
            ]
        # final convs
        sequence += [
            nn.Conv2d(ndf*nf_mult, ndf*nf_mult, kernel_size=kw, stride=1, padding=pw),
            nn.InstanceNorm2d(ndf*nf_mult, affine=False, eps=1e-5),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=pw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor, return_feats: bool = True):
        feats = []
        out = x
        for layer in self.model:
            out = layer(out)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(out)
        if return_feats:
            return out, feats
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    Three discriminators at different image scales (as in Pix2PixHD).
    """
    def __init__(self, in_nc: int, num_D: int = 3, ndf: int = 64, n_layers: int = 4):
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(in_nc, ndf, n_layers) for _ in range(num_D)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        results = []
        inp = x
        for i in range(self.num_D):
            out = self.discriminators[i](inp, return_feats=True)
            results.append(out)
            if i != self.num_D - 1:
                inp = self.downsample(inp)
        return results


# ---------------------------
# Losses: LSGAN, Feature Matching, KL
# ---------------------------

class GANLoss(nn.Module):
    """
    Least-Squares GAN loss (as used in Pix2pixHD).
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def _target(self, preds: torch.Tensor, is_real: bool) -> torch.Tensor:
        return torch.ones_like(preds) if is_real else torch.zeros_like(preds)

    def d_loss(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        loss_real = self.loss(pred_real, self._target(pred_real, True))
        loss_fake = self.loss(pred_fake, self._target(pred_fake, False))
        return 0.5 * (loss_real + loss_fake)

    def g_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        return self.loss(pred_fake, self._target(pred_fake, True))


def feature_matching_loss(feats_real: List[torch.Tensor],
                          feats_fake: List[torch.Tensor]) -> torch.Tensor:
    """
    L1 feature matching across layers.
    """
    loss = 0.0
    for fr, ff in zip(feats_real, feats_fake):
        loss += F.l1_loss(ff, fr.detach())
    return loss


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, sigma^2) || N(0, I) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    averaged over batch.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------
# Full Model Wrapper
# ---------------------------

class SSS_SPADE_GAN(nn.Module):
    def __init__(
        self,
        img_nc: int = 1,          # sonar grayscale by default
        seg_nc: int = 3,          # e.g., seabed / highlight / shadow
        z_dim: int = 256,
        ngf: int = 64,
        ndf: int = 64,
        num_D: int = 3,
        n_layers_D: int = 4,
        img_size: int = 256,
        fm_lambda: float = 10.0   # feature matching weight (λ)
    ):
        super().__init__()
        self.z_dim = z_dim
        self.seg_nc = seg_nc
        self.img_nc = img_nc
        self.fm_lambda = fm_lambda

        # Modules
        self.pdp = PixelDistributionPredictor(in_nc=img_nc, z_dim=z_dim)
        self.netG = SPADEGenerator(z_dim=z_dim, out_nc=img_nc, label_nc=seg_nc, ngf=ngf, out_size=img_size)
        # Discriminators take concat([image, seg_onehot]) as channels
        self.netD = MultiScaleDiscriminator(in_nc=img_nc + seg_nc, num_D=num_D, ndf=ndf, n_layers=n_layers_D)

        # Losses
        self.gan_loss = GANLoss()

        # Init
        self.apply(weights_init_normal)

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_G(self, seg_oh: torch.Tensor, real_img: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        If real_img provided: use PDP to sample z ~ N(μ, σ) from real image (training).
        Else: z ~ N(0, I) (inference).
        """
        if real_img is not None:
            mu, logvar = self.pdp(real_img)
            z = self.sample_z(mu, logvar)
            fake = self.netG(z, seg_oh)
            return {"fake": fake, "mu": mu, "logvar": logvar}
        else:
            z = torch.randn(seg_oh.size(0), self.z_dim, device=seg_oh.device)
            fake = self.netG(z, seg_oh)
            return {"fake": fake}

    def D_forward(self, img: torch.Tensor, seg_oh: torch.Tensor):
        x = torch.cat([img, seg_oh], dim=1)
        return self.netD(x)

    def compute_D_loss(self, real_img: torch.Tensor, fake_img: torch.Tensor, seg_oh: torch.Tensor) -> torch.Tensor:
        d_out_real = self.D_forward(real_img, seg_oh)
        d_out_fake = self.D_forward(fake_img.detach(), seg_oh)

        loss = 0.0
        for (pred_r, _feats_r), (pred_f, _feats_f) in zip(d_out_real, d_out_fake):
            loss += self.gan_loss.d_loss(pred_r, pred_f)
        return loss / len(d_out_real)

    def compute_G_loss(self, real_img: torch.Tensor, fake_img: torch.Tensor, seg_oh: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        d_out_fake = self.D_forward(fake_img, seg_oh)
        d_out_real = self.D_forward(real_img, seg_oh)

        loss_g_gan = 0.0
        loss_g_fm  = 0.0
        for (pred_r, feats_r), (pred_f, feats_f) in zip(d_out_real, d_out_fake):
            loss_g_gan += self.gan_loss.g_loss(pred_f)
            loss_g_fm  += feature_matching_loss(feats_r, feats_f)

        loss_g_gan /= len(d_out_fake)
        loss_g_fm  /= len(d_out_fake)

        loss = loss_g_gan + self.fm_lambda * loss_g_fm
        logs = {"G_gan": float(loss_g_gan.item()), "G_fm": float(loss_g_fm.item())}
        return loss, logs


# ---------------------------
# Training Skeleton
# ---------------------------

def train_one_epoch(
    model: SSS_SPADE_GAN,
    loader: DataLoader,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    kl_lambda: float = 1.0,
    grad_accum_steps: int = 1
):
    model.train()
    running = {"D":0.0, "G":0.0, "KL":0.0, "G_gan":0.0, "G_fm":0.0}
    for step, batch in enumerate(loader, 1):
        # Expect batch dict: {"image": [B, C, H, W], "seg": [B,H,W] or [B,K,H,W]}
        real = batch["image"].to(device)           # normalized to [-1,1]
        seg  = batch["seg"].to(device)
        seg_oh = one_hot(seg, num_classes)

        # ---- G forward
        outG = model.forward_G(seg_oh, real_img=real)
        fake = outG["fake"]
        mu, logvar = outG["mu"], outG["logvar"]

        # ---- D update
        opt_D.zero_grad(set_to_none=True)
        d_loss = model.compute_D_loss(real, fake, seg_oh)
        (d_loss / grad_accum_steps).backward()
        if step % grad_accum_steps == 0:
            opt_D.step()

        # ---- G update
        opt_G.zero_grad(set_to_none=True)
        g_loss, g_logs = model.compute_G_loss(real, fake, seg_oh)
        kl = kl_divergence_loss(mu, logvar) * kl_lambda
        total_g = g_loss + kl
        (total_g / grad_accum_steps).backward()
        if step % grad_accum_steps == 0:
            opt_G.step()

        running["D"] += d_loss.item()
        running["G"] += g_loss.item()
        running["KL"]+= kl.item()
        running["G_gan"] += g_logs["G_gan"]
        running["G_fm"]  += g_logs["G_fm"]

    n = len(loader)
    for k in running:
        running[k] /= max(n, 1)
    return running


@torch.no_grad()
def sample_inference(
    model: SSS_SPADE_GAN,
    seg_batch: torch.Tensor,        # [B, K, H, W] or [B,1,H,W] int labels
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    model.eval()
    seg_oh = one_hot(seg_batch.to(device), num_classes)
    out = model.forward_G(seg_oh, real_img=None)
    return out["fake"]  # [-1,1]