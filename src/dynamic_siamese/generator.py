"""Kernel generators: from a global vector (old) or feature map (spatial)."""
import torch.nn as nn


class KernelGenerator(nn.Module):
    """Global kernel generator: (B, in_dim) -> (B, out_c, in_c, k, k)."""

    def __init__(self, in_dim, out_channels, in_channels, kernel_size,
                 hidden_dim=128):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                out_channels * in_channels * kernel_size * kernel_size,
            ),
        )

    def forward(self, feat):
        if feat.dim() != 2:
            raise ValueError(
                "KernelGenerator expects (B, in_dim), got "
                + str(tuple(feat.shape))
            )
        w = self.net(feat)
        return w.view(
            -1, self.out_channels, self.in_channels,
            self.kernel_size, self.kernel_size,
        )


class SpatialKernelGenerator(nn.Module):
    """Spatial kernel generator: (B, context_ch, H, W) -> (B, C, k*k, H*W).

    Uses 1x1 convolutions so spatial resolution is preserved. The output
    is a per-position kernel for depthwise convolution in
    SpatiallyVaryingConv2d.
    """

    def __init__(self, context_ch, side_ch, kernel_size=3, hidden_dim=32):
        super().__init__()
        self.side_ch = side_ch
        self.kernel_size = kernel_size
        kk = kernel_size * kernel_size

        self.net = nn.Sequential(
            nn.Conv2d(context_ch, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, side_ch * kk, 1),
        )

    def forward(self, feat_map):
        if feat_map.dim() != 4:
            raise ValueError(
                "SpatialKernelGenerator expects (B, C, H, W), got "
                + str(tuple(feat_map.shape))
            )
        out = self.net(feat_map)  # (B, side_ch*k*k, H, W)
        B, _, H, W = out.shape
        kk = self.kernel_size * self.kernel_size
        out = out.view(B, self.side_ch, kk, H, W)
        out = out.reshape(B, self.side_ch, kk, H * W)
        return out
