import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========
# SOCA
# ========
class SPDConv(nn.Module):
    """Space-to-Depth Convolution for tiny targets."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.act(self.bn(self.conv(x)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, c1, c2=None, reduction=16, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c2 = c2 or c1
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(c1 // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c1 // reduction, 1), c1, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
        self.bn = nn.BatchNorm2d(c2) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        x = x * self.sa(res)
        return self.bn(self.proj(x)) if isinstance(self.proj, nn.Conv2d) else self.proj(x)
    
class RepBlock(nn.Module):
    """Simple Rep-style block (train-time version)."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))


class RepNCSPELAN4(nn.Module):
    """
    Simplified ELAN + CSP structure.
    """
    def __init__(self, c1, c2, n=2):
        super().__init__()
        c_ = c2 // 2

        self.cv1 = nn.Conv2d(c1, c_, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, bias=False)

        self.blocks = nn.Sequential(*[RepBlock(c_) for _ in range(n)])

        self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.act(self.bn(self.cv3(torch.cat((y1, y2), dim=1))))
    
class SPPELAN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c1 // 2

        self.cv1 = nn.Conv2d(c1, c_, 1, bias=False)

        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(9, 1, 4)
        self.pool3 = nn.MaxPool2d(13, 1, 6)

        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.cv1(x)
        y = torch.cat([x, self.pool1(x), self.pool2(x), self.pool3(x)], dim=1)
        return self.act(self.bn(self.cv2(y)))
    

# =======
# BES
# =======
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid() 

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.view(b * self.groups, -1, h, w) 
        x1 = self.conv1x1(nn.AdaptiveAvgPool2d(1)(group_x))
        x2 = self.conv3x3(group_x)
        out = self.sig(x1 * x2)
        return (out * group_x).view(b, c, h, w)

class BiFPN_Concat2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Concat2, self).__init__()
        self.w = nn.Parameter(torch.ones(len(c1), dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        
        self.projections = nn.ModuleList([
            nn.Conv2d(ch_in, c2, kernel_size=1, stride=1, padding=0) if ch_in != c2 else nn.Identity()
            for ch_in in c1
        ])

    def forward(self, x):
        w = F.relu(self.w)
        w = w / (torch.sum(w, dim=0) + self.epsilon)
        x_proj = [proj(feat) for proj, feat in zip(self.projections, x)]
        return (w.view(-1, 1, 1, 1, 1) * torch.stack(x_proj)).sum(dim=0)

# ==============
# combination
# ==============
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, n=1, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()

# ================
# YOLO11-SDC
# ================
class DRB(nn.Module):  # Dual Residual Block
    """Dual Residual Block with Large-Kernel and Dilated Convolutions."""
    def __init__(self, c1, c2, k=7, shortcut=True):
        super().__init__()
        # Path A: Large Kernel Conv
        self.lk_conv = nn.Conv2d(c1, c2, kernel_size=k, padding=k//2, groups=math.gcd(c1, c2), bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        
        # Path B: Dilated Convs (Simulated with two cascaded dilated 3x3 convs)
        self.d_conv1 = nn.Conv2d(c1, c2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.d_conv2 = nn.Conv2d(c2, c2, kernel_size=3, padding=3, dilation=3, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        self.act = nn.SiLU()
        self.shortcut = shortcut and c1 == c2
        
    def forward(self, x):
        # Path A
        y1 = self.bn1(self.lk_conv(x))
        # Path B
        y2 = self.bn2(self.d_conv2(self.d_conv1(x)))
        # Fusion
        out = self.act(y1 + y2)
        return out + x if self.shortcut else out

class C3k2_DRB(nn.Module):
    """C3-style architecture utilizing the DRB bottleneck."""
    # ADD 'n=1' to the initialization signature
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, c_, 1, bias=False)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
        # Wrap the inner bottleneck in a sequential loop to handle 'n' repeats internally
        self.m = nn.Sequential(*(DRB(c_, c_, shortcut=shortcut) for _ in range(n)))

    def forward(self, x):
        return self.act(self.bn(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))))

# =======================
# Spatial Fusion (SF)
# =======================
class SF(nn.Module):
    """Spatial Fusion block (replaces standard SPPF)."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, bias=False)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.act(self.bn1(self.cv1(x)))
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.act(self.bn2(self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))))

# =======================
# CGAF Sub-Attentions
# =======================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(y)

class PixelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.conv(x))

# ==========================================
# Cross-Granularity Attention Fusion (CGAF)
# ==========================================
class CGAF(nn.Module):  # Cross-Granularity Attention Fusion (CGAF)
    """Cross-Granularity Attention Fusion for feature maps from Backbone & Neck."""
    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, list):
            ch1, ch2 = c1[0], c1[1]
        else:
            ch1, ch2 = c1, c1
            
        # Align channels if backbone and neck feature maps differ
        self.align = nn.Conv2d(ch1, ch2, 1, bias=False) if ch1 != ch2 else nn.Identity()
        
        self.ca = ChannelAttention(ch2)
        self.sa = SpatialAttention()
        self.pa = PixelAttention(ch2)
        
        self.cv = nn.Conv2d(ch2, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # x is a list of two tensors [Backbone_Feature, Neck_Feature]
        x1, x2 = x[0], x[1]
        x1 = self.align(x1)
        
        # Align spatial dimensions (Upsamples the smaller map to match the larger map)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
            
        # Element-wise addition of the two feature maps
        fused = x1 + x2
        
        # Parallel CA and SA branches
        ca_out = self.ca(fused)
        sa_out = self.sa(fused)
        
        # Merge Attentions and pass through PA
        attn_fused = ca_out + sa_out
        pa_out = self.pa(attn_fused)
        
        return self.act(self.bn(self.cv(pa_out)))
