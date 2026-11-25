"""Full network architecture.

Two side-block variants:
  SideBlock          - old: single kernel per sample (global)
  SpatialSideBlock   - new: per-position kernels (--spatial)

Both use dual zero-init (side_scale = 0, kernel_scale = 0) so at
initialization the model is numerically equal to SiameseBaseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from .layers import ExternalKernelConv2d, SpatiallyVaryingConv2d
from .generator import KernelGenerator, SpatialKernelGenerator


class ResNetTower(nn.Module):
    """ResNet18 backbone with optional side-block between L3 and L4."""

    LAYER3_CH = 256
    LAYER4_CH = 512

    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        r = resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, side_block=None, side_context=None,
                return_features=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feats = self.layer3(x)
        if side_block is not None:
            feats = side_block(feats, side_context)
        x = self.layer4(feats)
        out = self.pool(x).flatten(1)
        if return_features:
            return out, feats
        return out

    def forward_features(self, x, level="l3"):
        """Return (pooled_output, feature_map_at_level).

        level: one of 'l1', 'l2', 'l3', 'l4'. side_block is not applied
        (used by HyperComparisonNet, which has no side-block).
        """
        layers = {
            "l1": self.layer1, "l2": self.layer2,
            "l3": self.layer3, "l4": self.layer4,
        }
        order = ["l1", "l2", "l3", "l4"]
        x = self.stem(x)
        feats = None
        for name in order:
            x = layers[name](x)
            if name == level:
                feats = x
        out = self.pool(x).flatten(1)
        return out, feats


class SideBlock(nn.Module):
    """Global-kernel side-branch (original design)."""

    def __init__(self, base_ch, side_ch, context_dim, generate_kernels=True,
                 gen_hidden_dim=32):
        super().__init__()
        self.generate_kernels = generate_kernels
        self.down = nn.Conv2d(base_ch, side_ch, 1, bias=False)
        self.bn_down = nn.BatchNorm2d(side_ch)
        self.conv = ExternalKernelConv2d(
            side_ch, side_ch, kernel_size=3, stride=1, padding=1,
            bias=False,
        )
        self.up = nn.Conv2d(side_ch, base_ch, 1, bias=False)
        self.bn_up = nn.BatchNorm2d(base_ch)

        self.side_scale = nn.Parameter(torch.zeros(1))
        self.base_kernel = nn.Parameter(torch.empty(side_ch, side_ch, 3, 3))
        nn.init.kaiming_uniform_(self.base_kernel, a=5 ** 0.5)

        if generate_kernels:
            self.kernel_gen = KernelGenerator(
                in_dim=context_dim,
                out_channels=side_ch,
                in_channels=side_ch,
                kernel_size=3,
                hidden_dim=gen_hidden_dim,
            )
            self.kernel_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, context):
        if self.generate_kernels:
            delta = self.kernel_gen(context)
            kernel = self.base_kernel.unsqueeze(0) + self.kernel_scale * delta
        else:
            kernel = self.base_kernel
        h = F.gelu(self.bn_down(self.down(x)))
        h = self.conv(h, kernel)
        h = self.bn_up(self.up(h))
        return x + self.side_scale * h


class SpatialSideBlock(nn.Module):
    """Spatially-varying side-branch.

    Generates one depthwise kernel per spatial position, conditioned on
    the tower-A layer3 feature map. Same dual zero-init as SideBlock.
    """

    def __init__(self, base_ch, side_ch, context_ch, generate_kernels=True,
                 gen_hidden_dim=32, kernel_size=3):
        super().__init__()
        self.generate_kernels = generate_kernels
        self.side_ch = side_ch
        self.kernel_size = kernel_size

        self.down = nn.Conv2d(base_ch, side_ch, 1, bias=False)
        self.bn_down = nn.BatchNorm2d(side_ch)
        self.spatial_conv = SpatiallyVaryingConv2d(
            side_ch, kernel_size=kernel_size, padding=kernel_size // 2,
        )
        self.up = nn.Conv2d(side_ch, base_ch, 1, bias=False)
        self.bn_up = nn.BatchNorm2d(base_ch)

        self.side_scale = nn.Parameter(torch.zeros(1))

        # Learnable base kernel, broadcast across positions and batch.
        self.base_kernel_global = nn.Parameter(
            torch.empty(side_ch, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.base_kernel_global, a=5 ** 0.5)

        if generate_kernels:
            self.kernel_gen = SpatialKernelGenerator(
                context_ch=context_ch,
                side_ch=side_ch,
                kernel_size=kernel_size,
                hidden_dim=gen_hidden_dim,
            )
            self.kernel_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, context):
        B, _, H, W = x.shape
        kk = self.kernel_size * self.kernel_size
        base = self.base_kernel_global.view(1, self.side_ch, kk, 1)

        if self.generate_kernels:
            delta = self.kernel_gen(context)  # (B, C, kk, H*W)
            kernel = base + self.kernel_scale * delta
        else:
            kernel = base.expand(B, -1, -1, H * W)

        h = F.gelu(self.bn_down(self.down(x)))
        h = self.spatial_conv(h, kernel)
        h = self.bn_up(self.up(h))
        return x + self.side_scale * h


class DynamicSiameseNet(nn.Module):
    def __init__(self, embedding_dim=128, generate_kernels=True,
                 pretrained=False, share_backbone=False, side_channels=64,
                 spatial=False):
        super().__init__()
        self.spatial = spatial
        self.tower_a = ResNetTower(pretrained=pretrained)
        if share_backbone:
            self.tower_b = self.tower_a
        else:
            self.tower_b = ResNetTower(pretrained=pretrained)

        if spatial:
            self.side_block = SpatialSideBlock(
                base_ch=ResNetTower.LAYER3_CH,
                side_ch=side_channels,
                context_ch=ResNetTower.LAYER3_CH,
                generate_kernels=generate_kernels,
            )
        else:
            self.side_block = SideBlock(
                base_ch=ResNetTower.LAYER3_CH,
                side_ch=side_channels,
                context_dim=ResNetTower.LAYER4_CH,
                generate_kernels=generate_kernels,
            )

        self.emb_head_a = nn.Linear(ResNetTower.LAYER4_CH, embedding_dim)
        if share_backbone:
            self.emb_head_b = self.emb_head_a
        else:
            self.emb_head_b = nn.Linear(ResNetTower.LAYER4_CH, embedding_dim)

    def forward(self, x1, x2):
        if self.spatial:
            z_a, feats_a = self.tower_a(x1, return_features=True)
            z_b = self.tower_b(
                x2, side_block=self.side_block, side_context=feats_a,
            )
        else:
            z_a = self.tower_a(x1)
            z_b = self.tower_b(
                x2, side_block=self.side_block, side_context=z_a,
            )
        e_a = F.normalize(self.emb_head_a(z_a), dim=-1)
        e_b = F.normalize(self.emb_head_b(z_b), dim=-1)
        return (e_a * e_b).sum(dim=-1)

    def predict_prob(self, x1, x2):
        """Probability of similarity in [0, 1]. Matches training loss."""
        return torch.sigmoid(self(x1, x2) * 5.0)


class SiameseBaseline(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=False):
        super().__init__()
        self.tower = ResNetTower(pretrained=pretrained)
        self.emb_head = nn.Linear(ResNetTower.LAYER4_CH, embedding_dim)

    def _encode(self, x):
        z = self.tower(x)
        return F.normalize(self.emb_head(z), dim=-1)

    def forward(self, x1, x2):
        e1 = self._encode(x1)
        e2 = self._encode(x2)
        return (e1 * e2).sum(dim=-1)

    def predict_prob(self, x1, x2):
        """Probability of similarity in [0, 1]. Matches training loss."""
        return torch.sigmoid(self(x1, x2) * 5.0)
