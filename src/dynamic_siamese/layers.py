"""Low-level PyTorch layers for dynamic kernel injection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalKernelConv2d(nn.Module):
    """Conv2d that either uses its own learnable kernel or accepts an
    external kernel at forward time.

    external_weight shape:
        (out_c, in_c, kH, kW)        - single kernel shared across batch
        (B, out_c, in_c, kH, kW)     - per-sample kernel (batched)

    Batched kernels use grouped convolution; no Python loop.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, external_weight=None):
        if external_weight is None:
            return F.conv2d(x, self.weight, self.bias,
                            stride=self.stride, padding=self.padding)

        if external_weight.dim() == 4:
            return F.conv2d(x, external_weight, self.bias,
                            stride=self.stride, padding=self.padding)

        if external_weight.dim() != 5:
            raise ValueError(
                "external_weight must be 4D or 5D, got "
                + str(tuple(external_weight.shape))
            )

        B, out_c, in_c, kH, kW = external_weight.shape
        if x.shape[0] != B:
            raise ValueError(
                "Batch size mismatch: input has " + str(x.shape[0])
                + ", external_weight has " + str(B)
            )

        x_reshaped = x.reshape(1, B * x.shape[1], *x.shape[2:])
        w_reshaped = external_weight.reshape(B * out_c, in_c, kH, kW)
        b = self.bias.repeat(B) if self.bias is not None else None

        out = F.conv2d(x_reshaped, w_reshaped, b,
                       stride=self.stride, padding=self.padding, groups=B)
        return out.reshape(B, out_c, *out.shape[2:])


class SpatiallyVaryingConv2d(nn.Module):
    """Depthwise conv with per-position kernels.

    Input:
        x:       (B, C, H, W)
        weights: (B, C, k*k, H*W)   one kernel per position
                 OR (1, C, k*k, H*W) OR (B, C, k*k, 1) — auto-broadcast

    Output:
        (B, C, H, W)   (same spatial size, stride=1, padding=k//2)

    Implementation: unfold + element-wise multiply + sum. Memory
    footprint ~3-4x the input tensor's, not O(C*k*k*H*W*batch).
    """

    def __init__(self, channels, kernel_size=3, padding=None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding if padding is not None else kernel_size // 2

    def forward(self, x, weights):
        B, C, H, W = x.shape
        k = self.kernel_size

        if weights.dim() != 4 or weights.shape[1] != C:
            raise ValueError(
                "weights must be (B, C, k*k, H*W), got "
                + str(tuple(weights.shape))
            )

        # Broadcast on batch and/or spatial.
        if weights.shape[0] == 1 and B > 1:
            weights = weights.expand(B, -1, -1, -1)
        if weights.shape[-1] == 1 and H * W > 1:
            weights = weights.expand(-1, -1, -1, H * W)

        patches = F.unfold(x, kernel_size=k, padding=self.padding)
        patches = patches.view(B, C, k * k, H * W)

        out = (patches * weights).sum(dim=2)
        return out.view(B, C, H, W)
