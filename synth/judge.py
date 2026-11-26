"""Object-vs-seabed judge: a binary classifier that answers "is there an object in this box?".

Used to verify generated objects. It is trained on the same real subset as the generators and calibrated
on the *unseen* val split, so its acceptance rate on real objects is known (see scripts/train_judge.py).
Crops are in the canonical orientation (shadow pointing right), like the generators' windows.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

SIZE = 64          # classifier input side
CONTEXT = 2.0      # crop side = CONTEXT * max(box w, h)
MIN_SIDE = 16


def crop(gray, box, starboard):
    """SIZE x SIZE float crop in [0, 1] around a box, canonical orientation; zero-padded at image borders."""
    x0, y0, x1, y1 = box
    side = int(max(MIN_SIDE, CONTEXT * max(x1 - x0, y1 - y0)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    h, w = gray.shape
    pad = side
    g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    c = g[cy - side // 2 + pad: cy - side // 2 + pad + side, cx - side // 2 + pad: cx - side // 2 + pad + side]
    c = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return c if starboard else c[:, ::-1].copy()


def build_model(pretrained=True):
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def to_tensor(crops):
    """List of SIZE x SIZE crops in [0,1] -> normalised 3-channel batch (ImageNet statistics)."""
    x = torch.tensor(np.stack(crops))[:, None].repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


class Judge:
    """Loads a trained judge; `prob(gray, box, starboard)` and `accept(...)` against the calibrated threshold."""

    def __init__(self, directory, device="cpu"):
        import json
        d = Path(directory)
        self.meta = json.loads((d / "meta.json").read_text())
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(d / "judge.pt", map_location=device))
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def prob(self, gray, box, starboard):
        x = to_tensor([crop(gray, box, starboard)]).to(self.device)
        return float(torch.sigmoid(self.model(x))[0, 0])

    def accept(self, gray, box, starboard):
        return self.prob(gray, box, starboard) >= self.meta["threshold"]
