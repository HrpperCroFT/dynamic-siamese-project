"""HyperComparisonNet: learned comparator via reference-conditioned kernels.

Architecture (as originally intended):
    Reference R  -> tower_a -> z_ref (B, 512), feats_ref (B, C, H, W)
                                │
                                └-> kernel_gen -> N kernels (B, N, out, in, k, k)
                                                    │
    Candidate C  -> tower_b -> feats_cand (B, C, H, W)
                                                    │
    ReferenceConditionedHead(feats_cand, kernels):
        N convs with reference-specific kernels -> N feature maps
        -> GAP -> concat -> MLP -> scalar
        -> optional cosine residual from a shared emb_head

    Similarity = sigmoid(head_out) or sigmoid(cos + alpha * head_out)

Two generator modes:
    conv : (B, C_ref, H_ref, W_ref) -> (B, N*out*in*k*k, 1, 1) -> reshape
    mlp  : z_ref (B, 512) -> FC -> (B, N*out*in*k*k) -> reshape

Zero-init:
    combine_alpha = 0  -> hyper mode starts exactly at baseline cosine
    head.out_scale = 1 -> head contributes when alpha > 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ResNetTower


class MultiKernelGenerator(nn.Module):
    """Produce N global conv kernels conditioned on reference features."""

    def __init__(self, context_dim, in_ch_ref, n_kernels, out_ch, in_ch,
                 kernel_size=3, mode="conv", hidden_dim=32):
        super().__init__()
        self.n_kernels = n_kernels
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.mode = mode

        total_out = n_kernels * out_ch * in_ch * kernel_size * kernel_size
        self.total_out = total_out

        if mode == "conv":
            # GAP BEFORE the final projection: prevents (B, total_out, H, W)
            # intermediate tensor which is 7.4 GB for B=32, l2 level.
            self.net = nn.Sequential(
                nn.Conv2d(in_ch_ref, 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, total_out),
            )
        elif mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, total_out),
            )
        else:
            raise ValueError("kernel_gen_mode must be 'conv' or 'mlp'")

    def forward(self, feats_ref, z_ref):
        """feats_ref: (B, C_ref, H, W); z_ref: (B, context_dim)."""
        if self.mode == "conv":
            out = self.net(feats_ref)                  # (B, total_out)
        else:
            out = self.net(z_ref)                      # (B, total_out)

        B = out.shape[0]
        return out.view(
            B, self.n_kernels, self.out_ch, self.in_ch,
            self.kernel_size, self.kernel_size,
        )


class ReferenceConditionedHead(nn.Module):
    """N convs with reference-specific kernels -> scalar similarity."""

    def __init__(self, in_ch, out_ch_per_kernel, n_kernels, kernel_size=3,
                 hidden_dim=128):
        super().__init__()
        self.n_kernels = n_kernels
        self.in_ch = in_ch
        self.out_ch_per_kernel = out_ch_per_kernel
        self.kernel_size = kernel_size
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(out_ch_per_kernel) for _ in range(n_kernels)
        ])
        feat_dim = n_kernels * out_ch_per_kernel
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, feats_cand, kernels):
        """feats_cand: (B, in_ch, H, W); kernels: (B, N, out, in, k, k)."""
        outs = []
        B, _, H, W = feats_cand.shape
        for i in range(self.n_kernels):
            # Per-example kernels via grouped convolution. There are no
            # unused fallback Conv2d weights in the adaptation parameter set.
            x = feats_cand.reshape(1, B * self.in_ch, H, W)
            weight = kernels[:, i].reshape(
                B * self.out_ch_per_kernel, self.in_ch,
                self.kernel_size, self.kernel_size,
            )
            h = F.conv2d(
                x, weight, padding=self.kernel_size // 2, groups=B,
            ).reshape(B, self.out_ch_per_kernel, H, W)
            h = F.gelu(self.bns[i](h))
            h = h.mean(dim=(2, 3))                         # GAP
            outs.append(h)
        h = torch.cat(outs, dim=1)                         # (B, N*out)
        return self.out_scale * self.mlp(h).squeeze(-1)    # (B,)


class HyperComparisonNet(nn.Module):
    """Two-tower network with a reference-conditioned learned comparator.

    Args:
        n_kernels:              number of kernels generated per reference
        feature_level:          'l1'|'l2'|'l3'|'l4' (which feature map the
                                head operates on; lower = higher resolution)
        kernel_gen_mode:        'conv' or 'mlp'
        out_ch_per_kernel:      channels per generated kernel
        use_cosine_residual:    if True, output = cos + alpha * head_out
                                with alpha zero-init (starts as baseline).
                                If False, output = head_out only.
        share_backbone:         share ResNet weights between towers
    """

    # in_channels for each ResNet18 feature level
    LEVEL_CH = {"l1": 64, "l2": 128, "l3": 256, "l4": 512}

    def __init__(self, n_kernels=4, feature_level="l2", kernel_gen_mode="conv",
                 out_ch_per_kernel=16, use_cosine_residual=True,
                 share_backbone=True, pretrained=False, embedding_dim=128,
                 embedding_only_ckpt=None, symmetric=False):
        super().__init__()
        self.feature_level = feature_level
        self.n_kernels = n_kernels
        self.use_cosine_residual = use_cosine_residual
        self.symmetric = symmetric

        self.tower_a = ResNetTower(pretrained=pretrained)
        self.tower_b = self.tower_a if share_backbone else ResNetTower(
            pretrained=pretrained,
        )

        in_ch = self.LEVEL_CH[feature_level]
        self.kernel_gen = MultiKernelGenerator(
            context_dim=ResNetTower.LAYER4_CH,
            in_ch_ref=in_ch,
            n_kernels=n_kernels,
            out_ch=out_ch_per_kernel,
            in_ch=in_ch,
            kernel_size=3,
            mode=kernel_gen_mode,
        )
        self.head = ReferenceConditionedHead(
            in_ch=in_ch,
            out_ch_per_kernel=out_ch_per_kernel,
            n_kernels=n_kernels,
            kernel_size=3,
        )

        if use_cosine_residual:
            self.emb_head = nn.Linear(ResNetTower.LAYER4_CH, embedding_dim)
            self.combine_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x_ref, x_cand):
        z_ref, feats_ref = self.tower_a.forward_features(
            x_ref, level=self.feature_level,
        )
        z_cand, feats_cand = self.tower_b.forward_features(
            x_cand, level=self.feature_level,
        )

        kernels = self.kernel_gen(feats_ref, z_ref)
        head_out = self.head(feats_cand, kernels)

        if self.symmetric:
            kernels_bwd = self.kernel_gen(feats_cand, z_cand)
            head_out_bwd = self.head(feats_ref, kernels_bwd)
            head_out = 0.5 * (head_out + head_out_bwd)

        if self.use_cosine_residual:
            e_ref = F.normalize(self.emb_head(z_ref), dim=-1)
            e_cand = F.normalize(self.emb_head(z_cand), dim=-1)
            cos = (e_ref * e_cand).sum(-1)
            return cos + self.combine_alpha * head_out
        return head_out

    def predict_prob(self, x1, x2):
        """Probability of similarity in [0, 1]. Matches training loss."""
        import torch as _t
        return _t.sigmoid(self(x1, x2) * 5.0)

    def adapt_params(self):
        """Parameters to train during few-shot adaptation."""
        params = list(self.head.parameters()) + list(self.kernel_gen.parameters())
        if self.use_cosine_residual:
            params.append(self.combine_alpha)
        return params

    def freeze_backbone(self):
        for p in self.tower_a.parameters():
            p.requires_grad = False
        if self.tower_b is not self.tower_a:
            for p in self.tower_b.parameters():
                p.requires_grad = False
        if self.use_cosine_residual:
            for p in self.emb_head.parameters():
                p.requires_grad = False
