
"""Multi-level HyperComparisonNet.

Reference generates global kernels at MULTIPLE conv levels simultaneously.
Each level: own KernelGen (from ref feat) + SideBlock (residual).

Zero-init side_scale per level → starts as baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from .layers import ExternalKernelConv2d


LEVEL_CH = {"l1": 64, "l2": 128, "l3": 256, "l4": 512}


class KernelGen(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, hidden=32):
        super().__init__()
        self.k = k
        self.out_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, out_ch * out_ch * k * k),
        )

    def forward(self, feat):
        B = feat.shape[0]
        return self.net(feat).view(B, self.out_ch, self.out_ch, self.k, self.k)


class MultiSideBlock(nn.Module):
    def __init__(self, base_ch, side_ch, context_ch, k=3, hidden=32):
        super().__init__()
        self.k = k
        self.down = nn.Conv2d(base_ch, side_ch, 1, bias=False)
        self.bn_down = nn.BatchNorm2d(side_ch)
        self.conv = ExternalKernelConv2d(side_ch, side_ch, kernel_size=k,
                                          stride=1, padding=k // 2, bias=False)
        self.up = nn.Conv2d(side_ch, base_ch, 1, bias=False)
        self.bn_up = nn.BatchNorm2d(base_ch)
        self.side_scale = nn.Parameter(torch.zeros(1))
        self.base_kernel = nn.Parameter(torch.empty(side_ch, side_ch, k, k))
        nn.init.kaiming_uniform_(self.base_kernel, a=5 ** 0.5)
        self.kernel_scale = nn.Parameter(torch.zeros(1))
        self.kernel_gen = KernelGen(context_ch, side_ch, k, hidden)

    def forward(self, x, context):
        delta = self.kernel_gen(context)
        kernel = self.base_kernel.unsqueeze(0) + self.kernel_scale * delta
        h = F.gelu(self.bn_down(self.down(x)))
        h = self.conv(h, kernel)
        h = self.bn_up(self.up(h))
        return x + self.side_scale * h


class HyperMultiNet(nn.Module):
    """Two-tower network. tower_a and tower_b share weights (ResNet18),
    but tower_b (candidate) has side_blocks injected at chosen levels."""

    def __init__(self, levels=("l2", "l3"), side_ch=32, k=3, hidden=32,
                 pretrained=False, embedding_dim=128):
        super().__init__()
        self.levels = tuple(levels)
        r = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.side_blocks = nn.ModuleDict()
        for lv in self.levels:
            ch = LEVEL_CH[lv]
            self.side_blocks[lv] = MultiSideBlock(ch, side_ch, ch, k, hidden)

        self.emb_head = nn.Linear(LEVEL_CH["l4"], embedding_dim)

    def _ref(self, x):
        h = self.stem(x)
        feats = {}
        h = self.layer1(h)
        if "l1" in self.levels: feats["l1"] = h
        h = self.layer2(h)
        if "l2" in self.levels: feats["l2"] = h
        h = self.layer3(h)
        if "l3" in self.levels: feats["l3"] = h
        h = self.layer4(h)
        if "l4" in self.levels: feats["l4"] = h
        return self.pool(h).flatten(1), feats

    def _cand(self, x, ref_feats):
        h = self.stem(x)
        h = self.layer1(h)
        if "l1" in self.levels: h = self.side_blocks["l1"](h, ref_feats["l1"])
        h = self.layer2(h)
        if "l2" in self.levels: h = self.side_blocks["l2"](h, ref_feats["l2"])
        h = self.layer3(h)
        if "l3" in self.levels: h = self.side_blocks["l3"](h, ref_feats["l3"])
        h = self.layer4(h)
        if "l4" in self.levels: h = self.side_blocks["l4"](h, ref_feats["l4"])
        return self.pool(h).flatten(1)

    def forward(self, x_ref, x_cand):
        z_ref, feats = self._ref(x_ref)
        z_cand = self._cand(x_cand, feats)
        e1 = F.normalize(self.emb_head(z_ref), dim=-1)
        e2 = F.normalize(self.emb_head(z_cand), dim=-1)
        return (e1 * e2).sum(-1)

    def adapt_params(self):
        # Each MultiSideBlock owns and uses its matching kernel generator.
        return list(self.side_blocks.parameters())
